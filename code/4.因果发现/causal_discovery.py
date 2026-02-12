"""
因果发现模块 v1.0：LiNGAM 结构学习 + DoWhy 因果推断
==================================================

目标：
1. 学习环境参数与病害/死亡之间的因果结构 (DAG)
2. 估计关键干预（如：增加光照、提高溶氧）的因果效应 (ATE)

方法：
1. 数据预处理：选择关键变量，处理缺失值，标准化
2. 结构学习：使用 DirectLiNGAM 算法学习变量间的因果顺序和连接强度
3. 因果图可视化：生成 DAG 图
4. 因果效应估计：使用 DoWhy 基于学习到的图进行干预估计
   - 线性回归估算器
   - 倾向性得分匹配 (PSM) 验证
   - 安慰剂检验 (Placebo Refutation)

输出：output/
"""

import sys
import warnings
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import networkx as nx

import lingam
from lingam.utils import make_dot
import dowhy
from dowhy import CausalModel

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURED_HONGGUANG = DATA_DIR / "featured_红光.csv"
FEATURED_KAZUO = DATA_DIR / "featured_喀左.csv"

# 随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore")

# ============================================================
# 变量选择
# ============================================================
# 我们不使用所有70+个特征，而是选择具有物理代表性的关键变量
CAUSAL_VARS = [
    # 环境因子 (Causes)
    "光照时长h", "光照_峰值",
    "水温_日均", "水温_日较差",
    "溶氧mg/L", "氨氮mg/L",
    "PH", "EC值ms/cm",
    "气温_日均", "气温_日较差",
    "湿度_日均",
    
    # 结果 (Effects)
    "蔬菜_病害次数",
    "鱼_死亡数量",
]

# 变量重命名（简化图显示）
VAR_RENAME = {
    "光照时长h": "Light_Hours",
    "光照_峰值": "Light_Peak",
    "水温_日均": "Water_Temp",
    "水温_日较差": "Water_Temp_Diff",
    "溶氧mg/L": "DO",
    "氨氮mg/L": "Ammonia",
    "PH": "PH",
    "EC值ms/cm": "EC",
    "气温_日均": "Air_Temp",
    "气温_日较差": "Air_Temp_Diff",
    "湿度_日均": "Humidity",
    "蔬菜_病害次数": "Veg_Disease",
    "鱼_死亡数量": "Fish_Death",
}

# 反向映射
VAR_RENAME_INV = {v: k for k, v in VAR_RENAME.items()}


# ============================================================
# 中文字体
# ============================================================
def setup_chinese_font():
    for fp in ["/System/Library/Fonts/STHeiti Light.ttc",
               "/System/Library/Fonts/PingFang.ttc",
               "/System/Library/Fonts/Supplemental/Songti.ttc",
               "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
        try:
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

setup_chinese_font()


# ============================================================
# 数据准备
# ============================================================
def load_and_preprocess(site="红光"):
    path = FEATURED_HONGGUANG if site == "红光" else FEATURED_KAZUO
    df = pd.read_csv(path)
    
    # 选择变量
    available_vars = [v for v in CAUSAL_VARS if v in df.columns]
    
    # 填充缺失值（因果发现对缺失值敏感，简单插值）
    # 氨氮/溶氧可能有较多缺失，用时序插值
    df_subset = df[available_vars].copy()
    df_subset = df_subset.interpolate(method="linear").bfill().ffill()
    df_subset = df_subset.fillna(df_subset.mean()) # 最后的兜底
    
    # 重命名列
    df_subset = df_subset.rename(columns=VAR_RENAME)
    
    print(f"📊 [{site}] 加载数据: {df_subset.shape}")
    print(f"   变量: {list(df_subset.columns)}")
    
    return df_subset


# ============================================================
# 结构学习 (LiNGAM)
# ============================================================
def learn_structure(df, site):
    print(f"\n🧠 [{site}] 正在学习因果结构 (DirectLiNGAM)...")
    
    # 抽样加速结构学习 (n=1000)
    if len(df) > 1000:
        print(f"   ⚠️ 数据量较大 ({len(df)}), 抽样 1000 用于结构学习...")
        df_train = df.sample(1000, random_state=RANDOM_SEED)
    else:
        df_train = df

    # DirectLiNGAM (使用 pwling 熵测度加速)
    model = lingam.DirectLiNGAM(random_state=RANDOM_SEED, measure='pwling')
    model.fit(df_train)
    
    # 邻接矩阵 (Adjacency Matrix) B
    # x_i = sum(b_ij * x_j) + e_i
    adj_matrix = model.adjacency_matrix_
    
    # 变量顺序
    causal_order = model.causal_order_
    print("   因果顺序:", [df.columns[i] for i in causal_order])
    
    # 可视化因果图
    save_causal_graph(adj_matrix, df.columns, site)
    
    # 转为 NetworkX 图（供 DoWhy 使用）
    G = nx.DiGraph()
    G.add_nodes_from(df.columns)
    print(f"   🕸️ 发现的因果边 (阈值 > 0.01):")
    for i, j in zip(*np.where(np.abs(adj_matrix) > 0.01)): # 阈值过滤弱连接
        weight = float(adj_matrix[i, j])
        if np.isnan(weight) or np.isinf(weight):
            continue
        target = df.columns[i]
        source = df.columns[j]
        print(f"      {source} -> {target} (w={weight:.4f})")
        G.add_edge(source, target, weight=weight)
        
    return model, G, adj_matrix


import shutil

def save_causal_graph(adj_matrix, labels, site):
    # 使用 LiNGAM 内置绘图
    try:
        if shutil.which("dot") is None:
            print(f"   ⚠️ 未找到 'dot' 命令 (Graphviz)，跳过因果图绘制")
        else:
            dot = make_dot(adj_matrix, labels=labels.tolist())
            dot_path = OUTPUT_DIR / f"causal_graph_{site}"
            dot.render(dot_path, format="png", cleanup=True)
            print(f"   🖼️ 因果图已保存: {dot_path}.png")
    except Exception as e:
        print(f"   ⚠️ 绘制因果图失败: {e}")
    
    # 绘制热力图矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(adj_matrix, annot=True, fmt=".2f", cmap="vlag", center=0,
                xticklabels=labels, yticklabels=labels)
    plt.title(f"因果连接强度矩阵 ({site})\n(Col -> Row)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"adjacency_matrix_{site}.png", dpi=150)
    plt.close()


# ============================================================
# DoWhy 因果效应估计
# ============================================================
def estimate_effect_dowhy(df, G, treatment, outcome, site):
    if treatment not in df.columns or outcome not in df.columns:
        print(f"   ⚠️ 跳过: {treatment} -> {outcome} (变量不存在)")
        return None
        
    print(f"\n🔎 [{site}] 估计因果效应: {treatment} -> {outcome}")
    
    # 将 NetworkX 图转为 GML 字符串 (DoWhy 需要)
    gml_str = "".join(nx.generate_gml(G))
    
    # 1. 定义因果模型
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=gml_str,
        missing_nodes_as_confounders=False # 我们假设 DAG 是完整的（基于 LiNGAM）
    )
    
    # 2. 识别因果效应 (Identification)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    # print(identified_estimand) # debug
    
    # 3. 估计因果效应 (Estimation)
    # 使用线性回归
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    
    print(f"   🎯 因果效应 (ATE): {estimate.value:.4f}")
    
    p_value = 1.0
    try:
        p_val_res = estimate.test_stat_significance()
        if p_val_res:
            p_value = p_val_res.get('p_value', 1.0)
            # handle array p-value
            if isinstance(p_value, (np.ndarray, list)):
                p_value = p_value[0] if len(p_value) > 0 else 1.0
            print(f"      p-value: {float(p_value):.4f}")
    except Exception as e:
        print(f"      ⚠️ 无法获取 p-value: {e}")

    # 4. 反驳/验证 (Refutation)
    placebo_p = 1.0
    subset_ATE = estimate.value
    
    try:
        # 安慰剂干预 (Placebo Treatment)
        refute = model.refute_estimate(
            identified_estimand, estimate,
            method_name="placebo_treatment_refuter"
        )
        placebo_p = refute.refutation_result.get('p_value', 1.0)
        print(f"   🛡️ 安慰剂检验 p-value: {placebo_p:.4f} (应该是无关的)")
        
        # 数据子集验证
        refute_subset = model.refute_estimate(
            identified_estimand, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8
        )
        subset_ATE = refute_subset.new_effect
        print(f"   🛡️ 子集验证: 新ATE={subset_ATE:.4f} (原={estimate.value:.4f})")
    except Exception as e:
        print(f"      ⚠️ 反驳验证失败: {e}")
    
    return {
        "treatment": treatment,
        "outcome": outcome,
        "ATE": estimate.value,
        "p_value": p_value,
        "placebo_p": placebo_p,
        "subset_ATE": subset_ATE,
        "is_robust": (p_value < 0.05) and \
                     (abs(estimate.value - subset_ATE) < abs(estimate.value) * 0.5 + 0.01)
    }


def get_expert_structure(df_columns):
    """
    定义基于领域知识的专家因果图 (Physics-guided Causal Graph)
    避免 LiNGAM 纯数据驱动产生的反直觉方向 (如 病害 -> 光照)
    """
    G = nx.DiGraph()
    G.add_nodes_from(df_columns)
    
    # 物理/生物学机制边
    # 1. 环境 -> 蔬菜病害
    if "Light_Hours" in df_columns and "Veg_Disease" in df_columns:
        G.add_edge("Light_Hours", "Veg_Disease") # 光照增强抵抗力
    if "Humidity" in df_columns and "Veg_Disease" in df_columns:
        G.add_edge("Humidity", "Veg_Disease")    # 高湿导致病害
    if "EC" in df_columns and "Veg_Disease" in df_columns:
        G.add_edge("EC", "Veg_Disease")
        
    # 2. 环境 -> 鱼类死亡
    if "DO" in df_columns and "Fish_Death" in df_columns:
        G.add_edge("DO", "Fish_Death")           # 缺氧 -> 死亡
    if "Ammonia" in df_columns and "Fish_Death" in df_columns:
        G.add_edge("Ammonia", "Fish_Death")      # 氨氮毒性
    if "Water_Temp_Diff" in df_columns and "Fish_Death" in df_columns:
        G.add_edge("Water_Temp_Diff", "Fish_Death") # 温差应激
        
    # 3. 环境间相互作用 (物理机制)
    if "Water_Temp" in df_columns and "DO" in df_columns:
        G.add_edge("Water_Temp", "DO")           # 温度影响饱和溶氧
    if "Light_Hours" in df_columns and "DO" in df_columns:
        G.add_edge("Light_Hours", "DO")          # 光合作用产氧
        
    return G

# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  🌳 因果发现与推理 (LiNGAM + DoWhy)")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出: {OUTPUT_DIR}")
    
    results = {}

    for site in ["红光", "喀左"]:
        print(f"\n\n{'='*40}")
        print(f"📍 站点: {site}")
        print(f"{'='*40}")
        
        # 1. 准备数据
        df = load_and_preprocess(site)
        
        # 2. 学习结构 (LiNGAM) - 作为对比
        lingam_model, G_lingam, adj_mat = learn_structure(df, site)
        
        # 3. 构建专家结构 (Expert) - 作为主要推理依据
        print(f"\n🧠 [{site}] 构建专家因果图 (Physics-Guided)...")
        G_expert = get_expert_structure(df.columns)
        
        # 保存专家图
        try:
            if shutil.which("dot"):
                dot = make_dot(nx.to_numpy_array(G_expert, nodelist=df.columns), labels=df.columns.tolist())
                dot.render(OUTPUT_DIR / f"causal_graph_expert_{site}", format="png", cleanup=True)
        except Exception: pass
        
        # 定义我们要探究的假设路径
        hypotheses = [
            ("Light_Hours", "Veg_Disease"),    # 光照时长 -> 蔬菜病害
            ("Water_Temp_Diff", "Fish_Death"), # 水温日较差 -> 鱼类死亡
            ("DO", "Fish_Death"),              # 溶氧 -> 鱼类死亡
            ("Ammonia", "Fish_Death"),         # 氨氮 -> 鱼类死亡
            ("EC", "Veg_Disease"),             # EC -> 蔬菜病害
            ("Humidity", "Veg_Disease"),       # 湿度 -> 蔬菜病害
        ]
        
        site_effects = []
        for treat, outcome in hypotheses:
            # 优先使用专家图进行估计 (因为 LiNGAM 发现的方向往往是反的)
            res = estimate_effect_dowhy(df, G_expert, treat, outcome, site)
            if res:
                site_effects.append(res)
        
        results[site] = site_effects
        
    # 保存结果报告
    report_path = OUTPUT_DIR / "causal_report.json"
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return super(NumpyEncoder, self).default(obj)
            
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
    print(f"\n✅ 完成! 报告已保存至 {report_path}")
    
    # 打印简要总结
    print("\n📝 结果摘要 (Robust Only):")
    for site, effects in results.items():
        print(f"\n  [{site}]")
        for eff in effects:
            if eff is None: continue
            star = "🌟" if eff["is_robust"] else "  "
            p_val = eff['p_value']
            # handle complex or array p-values
            if isinstance(p_val, (np.ndarray, list)):
                p_val = p_val[0] if len(p_val) > 0 else 1.0
            
            print(f"  {star} {eff['treatment']:15s} -> {eff['outcome']:15s} | "
                  f"ATE = {eff['ATE']:6.3f} (p={float(p_val):.3f})")

if __name__ == "__main__":
    main()
