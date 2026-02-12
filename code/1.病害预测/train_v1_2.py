"""
病害预测 v1.2：PINN 反事实合成 + 迁移学习微调
==============================================
在 v1.1（物理特征 + 域自适应）基础上新增两大改进：

  A. PINN 反事实合成：利用物理方程生成"高温缺氧"致死场景的合成正样本，
     扩充红光训练集中与喀左死亡模式相似的极端样本。
  B. 两阶段迁移学习：
     阶段1 — 在红光(+合成)数据上预训练 XGBoost；
     阶段2 — 取喀左 10%（分层抽样）数据微调，剩余 90% 作为测试。

输出目录：output_v1.2/
"""
import sys
import warnings
import json
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config import (
    FEATURED_HONGGUANG, FEATURED_KAZUO,
    META_COLS, LABEL_COLS, CUMULATIVE_LABEL_COLS,
    CV_FOLDS, SHAP_TOP_N, RANDOM_SEED,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 路径与常量
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output_v1.2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
V1_OUTPUT = SCRIPT_DIR / "output"
V11_OUTPUT = SCRIPT_DIR / "output_v1.1"

# PINN 物理参数
PINN_PARAMS = {
    "K_La": 1.35815, "R_fish_base": 1.50715, "alpha_T": 0.0338,
    "R_bio": 0.751, "P_photo_rate": 0.06935, "T_ref": 25.0,
}

TASKS = {
    "蔬菜病害": {"label_col": "蔬菜_病害次数", "description": "蔬菜是否发生病害（二分类）"},
    "鱼类死亡": {"label_col": "鱼_死亡数量", "description": "鱼类是否发生死亡（二分类）"},
}

# 域偏移特征
DOMAIN_SHIFT_FEATURES = [
    "能耗km/h", "种植床1液位上限距种植床表面距离cm",
    "种植床2液位上限距种植床表面距离cm",
]

# 阶段1（预训练）：较弱正则化，多学红光知识
XGBOOST_PRETRAIN = {
    "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": RANDOM_SEED, "n_jobs": -1, "eval_metric": "logloss",
    "early_stopping_rounds": 30,
}

# 阶段2（微调）：更强正则化，学喀左轻量差异
XGBOOST_FINETUNE = {
    "n_estimators": 150, "max_depth": 3, "learning_rate": 0.02,
    "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 15,
    "gamma": 0.5, "reg_alpha": 1.0, "reg_lambda": 5.0,
    "random_state": RANDOM_SEED, "n_jobs": -1, "eval_metric": "logloss",
    "early_stopping_rounds": 20,
}

# 喀左用于微调的比例
FINETUNE_RATIO = 0.10


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
# 物理特征（同 v1.1）
# ============================================================
def do_saturation(T):
    return 14.62 - 0.3898 * T + 0.006969 * T**2 - 5.897e-5 * T**3


def compute_physics_features(df):
    p = PINN_PARAMS
    T = df["水温_日均"].fillna(25.0)
    DO = df["溶氧mg/L"].fillna(df["溶氧mg/L"].median())
    light = df["光照时长h"].fillna(0.0)
    DO_sat = do_saturation(T)
    out = pd.DataFrame(index=df.index)
    out["pinn_DO_deficit"] = DO_sat - DO
    out["pinn_R_fish_T"] = p["R_fish_base"] * (1 + p["alpha_T"] * (T - p["T_ref"]))
    out["pinn_reaeration"] = p["K_La"] * (DO_sat - DO)
    out["pinn_P_photo"] = p["P_photo_rate"] * light
    out["pinn_oxygen_stress"] = (out["pinn_R_fish_T"] + p["R_bio"]
                                  - out["pinn_reaeration"] - out["pinn_P_photo"])
    out["pinn_DO_margin"] = DO - 2.0
    out["pinn_DO_sat_ratio"] = DO / DO_sat.clip(lower=1.0)
    return out


def compute_temporal_features(df):
    out = pd.DataFrame(index=df.index)
    T = df["水温_日均"].fillna(25.0)
    T_air = df["气温_日均"].fillna(20.0)
    DO = df["溶氧mg/L"].fillna(df["溶氧mg/L"].median())
    roll3 = df.get("水温_日均_roll3d_mean")
    out["trend_水温_3d"] = (T - roll3.fillna(T)) if roll3 is not None else 0.0
    do_ch = df.get("溶氧_变化率")
    out["DO_volatility"] = do_ch.fillna(0).abs() if do_ch is not None else 0.0
    out["水气温_交互"] = T * T_air / 100.0
    out["水温_偏离适宜"] = np.where(T > 28, T - 28, np.where(T < 18, 18 - T, 0))
    out["高温天数_7d"] = (T > 28).astype(float).rolling(7, min_periods=1).sum().values
    out["低氧天数_7d"] = (DO < 5).astype(float).rolling(7, min_periods=1).sum().values
    roll3_std = df.get("水温_日均_roll3d_std")
    out["水温波动_加速度"] = roll3_std.fillna(0).diff().fillna(0) if roll3_std is not None else 0.0
    return out


# ============================================================
# PINN 反事实合成
# ============================================================
def generate_counterfactual_samples(df_source, label_col, n_synthetic=500):
    """
    利用 PINN 物理方程生成合成"高温缺氧致死"正样本。

    策略：
      1. 从红光数据中选取实际死亡事件发生时的样本作为模板
      2. 对模板进行物理一致的扰动：
         - 水温升高 1~5℃（模拟极端高温）
         - 溶氧按物理方程下降（DO_sat 降低 + 耗氧增加）
         - 光照时长随机缩短（模拟阴天/设备故障）
      3. 标记为正样本（有死亡）
      4. 对非死亡样本也施加物理极端条件作为补充

    Returns:
        df_synthetic: 合成样本 DataFrame（与原始同结构）
    """
    p = PINN_PARAMS
    rng = np.random.RandomState(RANDOM_SEED)

    y = (df_source[label_col] > 0).astype(int)

    # 模板1：从实际死亡事件中采样
    death_mask = y == 1
    if death_mask.sum() > 0:
        templates_death = df_source[death_mask].sample(
            n=min(n_synthetic // 2, death_mask.sum()),
            replace=True, random_state=RANDOM_SEED
        ).copy()
    else:
        templates_death = pd.DataFrame()

    # 模板2：从高水温的非死亡事件中采样（模拟"差一点就死了"场景）
    high_temp_mask = (y == 0) & (df_source["水温_日均"] > df_source["水温_日均"].quantile(0.75))
    n_from_negative = n_synthetic - len(templates_death)
    if high_temp_mask.sum() > 0 and n_from_negative > 0:
        templates_neg = df_source[high_temp_mask].sample(
            n=min(n_from_negative, high_temp_mask.sum()),
            replace=True, random_state=RANDOM_SEED + 1
        ).copy()
    else:
        templates_neg = pd.DataFrame()

    templates = pd.concat([templates_death, templates_neg], ignore_index=True)
    if len(templates) == 0:
        print("  ⚠️ 无法生成合成样本（无模板）")
        return pd.DataFrame()

    print(f"  🧪 合成模板: {len(templates_death)} 来自真实死亡, "
          f"{len(templates_neg)} 来自高温非死亡")

    # 物理一致扰动
    delta_T = rng.uniform(1.0, 5.0, size=len(templates))    # 升温 1~5℃
    light_factor = rng.uniform(0.3, 0.8, size=len(templates))  # 光照缩短

    syn = templates.copy()

    # 水温提升
    syn["水温_日均"] = syn["水温_日均"] + delta_T
    syn["最高水温℃"] = syn["最高水温℃"] + delta_T
    syn["最低水温℃"] = syn["最低水温℃"] + delta_T * 0.5

    # 光照缩短
    syn["光照时长h"] = syn["光照时长h"] * light_factor

    # 溶氧按物理方程校正
    T_new = syn["水温_日均"].values
    DO_sat_new = do_saturation(T_new)
    R_fish_new = p["R_fish_base"] * (1 + p["alpha_T"] * (T_new - p["T_ref"]))
    P_photo_new = p["P_photo_rate"] * syn["光照时长h"].values

    # 新 DO = DO_sat - (R_fish + R_bio - P_photo) / K_La
    # 这是稳态近似：dDO/dt ≈ 0 时的平衡点
    DO_equilibrium = DO_sat_new - (R_fish_new + p["R_bio"] - P_photo_new) / p["K_La"]
    DO_equilibrium = np.clip(DO_equilibrium, 0.5, DO_sat_new)  # 物理约束

    if "溶氧mg/L" in syn.columns:
        # 实际 DO 取当前值和平衡值的较低者（恶化）
        syn["溶氧mg/L"] = np.minimum(syn["溶氧mg/L"].values, DO_equilibrium)
        # 给一些随机扰动
        syn["溶氧mg/L"] = syn["溶氧mg/L"] - rng.uniform(0, 1.5, size=len(syn))
        syn["溶氧mg/L"] = syn["溶氧mg/L"].clip(lower=0.5)

    # 标记为正样本
    syn[label_col] = 1

    # 添加标记列
    syn["_is_synthetic"] = True

    # 滚动特征就直接继承模板的（近似合理）
    print(f"  ✅ 生成 {len(syn)} 个 PINN 反事实合成正样本")
    print(f"     水温: {syn['水温_日均'].mean():.1f} ± {syn['水温_日均'].std():.1f}℃"
          f"  (原始: {templates['水温_日均'].mean():.1f}℃)")
    if "溶氧mg/L" in syn.columns:
        print(f"     溶氧: {syn['溶氧mg/L'].mean():.1f} ± {syn['溶氧mg/L'].std():.1f} mg/L"
              f"  (原始: {templates['溶氧mg/L'].mean():.1f} mg/L)")

    return syn


# ============================================================
# 数据加载 + 特征增强 + 合成数据
# ============================================================
def load_all_data(task_config, use_synthetic=True, n_synthetic=500):
    """
    加载并增强全部数据。

    Returns:
        df_hg_full: 红光全集（含合成）+ 增强特征
        df_kz_full: 喀左全集 + 增强特征
        feature_names: 特征名
    """
    label_col = task_config["label_col"]

    df_hg = pd.read_csv(FEATURED_HONGGUANG, parse_dates=["日期"])
    df_kz = pd.read_csv(FEATURED_KAZUO, parse_dates=["日期"])
    print(f"[红光] {df_hg.shape[0]} 行 | [喀左] {df_kz.shape[0]} 行")

    # ====== PINN 反事实合成 ======
    if use_synthetic:
        print(f"\n🧬 PINN 反事实合成正样本...")
        syn = generate_counterfactual_samples(df_hg, label_col, n_synthetic)
        if len(syn) > 0:
            # 确保合成数据有相同列
            for c in df_hg.columns:
                if c not in syn.columns:
                    syn[c] = 0
            syn = syn[df_hg.columns.tolist() + ["_is_synthetic"]]
            df_hg["_is_synthetic"] = False
            df_hg = pd.concat([df_hg, syn[df_hg.columns]], ignore_index=True)
            print(f"  红光总样本: {len(df_hg)} (含 {len(syn)} 合成)")

    # ====== 增加物理 + 时序特征 ======
    for name, df in [("红光", df_hg), ("喀左", df_kz)]:
        phys = compute_physics_features(df)
        temp = compute_temporal_features(df)
        for c in phys.columns:
            df[c] = phys[c].values
        for c in temp.columns:
            df[c] = temp[c].values

    print(f"  ✅ +{len(compute_physics_features(df_hg).columns)} 物理 "
          f"+{len(compute_temporal_features(df_hg).columns)} 时序特征")

    # ====== 确定特征列 ======
    exclude = set(META_COLS + LABEL_COLS + CUMULATIVE_LABEL_COLS
                  + DOMAIN_SHIFT_FEATURES + ["_is_synthetic", "_is_augmented", "_scenario"])
    feature_cols = [c for c in df_hg.columns if c not in exclude]

    # 对齐列
    for c in feature_cols:
        if c not in df_kz.columns:
            df_kz[c] = 0

    return df_hg, df_kz, feature_cols


def prepare_Xy(df, feature_cols, label_col):
    """提取 X, y 并处理缺失"""
    X = df[feature_cols].copy()
    y = (df[label_col] > 0).astype(int)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(0)
    return X, y


# ============================================================
# 域自适应权重（同 v1.1）
# ============================================================
def compute_domain_weights(X_src, X_tgt):
    common = [c for c in X_src.columns if c in X_tgt.columns]
    scaler = StandardScaler()
    s = scaler.fit_transform(X_src[common].fillna(0))
    t = scaler.transform(X_tgt[common].fillna(0))
    n_comp = min(10, s.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    sp = pca.fit_transform(s)
    tp = pca.transform(t)
    centroid = tp.mean(axis=0)
    dist = np.sqrt(((sp - centroid) ** 2).sum(axis=1))
    sigma = np.median(dist) + 1e-8
    w = np.exp(-0.5 * (dist / sigma) ** 2)
    w = w / w.mean()
    w = np.clip(w, 0.1, 5.0)
    return w


# ============================================================
# 阈值搜索
# ============================================================
def find_optimal_threshold(y_true, y_prob):
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.05, 0.9, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


# ============================================================
# 评估
# ============================================================
def evaluate(model, X, y, name, task_name, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    m = {
        "数据集": name, "任务": task_name, "阈值": threshold,
        "样本数": len(y), "正样本数": int(y.sum()),
        "正样本比例": f"{y.mean()*100:.1f}%",
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1-Score": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else 0,
        "AP": average_precision_score(y, y_prob) if y.nunique() > 1 else 0,
    }

    print(f"\n{'─'*55}")
    print(f"📈 {name} ({task_name}) | thr={threshold:.2f}")
    print(f"{'─'*55}")
    for k, v in m.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")

    cm = confusion_matrix(y, y_pred)
    print(f"\n  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"[v1.2] {task_name} — {name}", fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["无", "有"], yticklabels=["无", "有"])
    axes[0].set_xlabel("预测"); axes[0].set_ylabel("实际"); axes[0].set_title("混淆矩阵")

    if y.nunique() > 1:
        fpr, tpr, _ = roc_curve(y, y_prob)
        axes[1].plot(fpr, tpr, "b-", lw=2, label=f"AUC={m['ROC-AUC']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC"); axes[1].legend()

        prec_arr, rec_arr, _ = precision_recall_curve(y, y_prob)
        axes[2].plot(rec_arr, prec_arr, "r-", lw=2, label=f"AP={m['AP']:.4f}")
        axes[2].axhline(y=y.mean(), color="gray", ls="--", alpha=0.5, label="随机")
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision"); axes[2].set_title("PR"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"eval_{task_name}_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return m


# ============================================================
# SHAP 分析
# ============================================================
def shap_analysis(model, X, feature_names, task_name, dataset_name=""):
    explainer = shap.TreeExplainer(model)
    X_s = X.sample(min(5000, len(X)), random_state=RANDOM_SEED) if len(X) > 5000 else X
    sv = explainer.shap_values(X_s)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(sv, X_s, feature_names=feature_names, max_display=SHAP_TOP_N, show=False)
    plt.title(f"[v1.2] {task_name} — SHAP (Top {SHAP_TOP_N})", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_summary_{task_name}_{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_s, feature_names=feature_names, plot_type="bar", max_display=SHAP_TOP_N, show=False)
    plt.title(f"[v1.2] {task_name} — SHAP 均值", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_bar_{task_name}_{dataset_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"特征": feature_names, "SHAP均值": mean_abs}).sort_values("SHAP均值", ascending=False).reset_index(drop=True)
    imp.index += 1; imp.index.name = "排名"
    imp.to_csv(OUTPUT_DIR / f"shap_importance_{task_name}_{dataset_name}.csv")

    print(f"\n  📋 Top 10:")
    print(imp.head(10).to_string())
    return imp


# ============================================================
# 加载历史报告
# ============================================================
def load_prev_report(task_name, version, output_dir):
    patterns = [f"report_{task_name}.json", f"report_{task_name}_{version}.json"]
    for pat in patterns:
        p = output_dir / pat
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


# ============================================================
# 主流程：一个任务
# ============================================================
def run_task(task_name, task_config, df_hg, df_kz, feature_cols):
    label_col = task_config["label_col"]

    print(f"\n{'#'*60}")
    print(f"## [v1.2] {task_name} — {task_config['description']}")
    print(f"{'#'*60}")

    # ====== 准备数据 ======
    X_hg, y_hg = prepare_Xy(df_hg, feature_cols, label_col)
    X_kz, y_kz = prepare_Xy(df_kz, feature_cols, label_col)

    print(f"\n  红光: {len(X_hg)} 样本, 正={y_hg.sum()} ({y_hg.mean()*100:.1f}%)")
    print(f"  喀左: {len(X_kz)} 样本, 正={y_kz.sum()} ({y_kz.mean()*100:.1f}%)")

    # ====== 喀左分层拆分：10% 微调 + 90% 测试 ======
    if y_kz.sum() >= 5:
        X_kz_ft, X_kz_test, y_kz_ft, y_kz_test = train_test_split(
            X_kz, y_kz, test_size=1.0 - FINETUNE_RATIO,
            random_state=RANDOM_SEED, stratify=y_kz
        )
    else:
        # 正样本太少，无法分层抽样
        X_kz_ft = X_kz.sample(frac=FINETUNE_RATIO, random_state=RANDOM_SEED)
        y_kz_ft = y_kz.loc[X_kz_ft.index]
        X_kz_test = X_kz.drop(X_kz_ft.index)
        y_kz_test = y_kz.drop(y_kz_ft.index)

    print(f"\n  📌 迁移学习拆分:")
    print(f"    喀左微调集: {len(X_kz_ft)} 样本, 正={y_kz_ft.sum()}")
    print(f"    喀左测试集: {len(X_kz_test)} 样本, 正={y_kz_test.sum()}")

    # ====== 红光拆分训练/验证 ======
    X_hg_train, X_hg_val, y_hg_train, y_hg_val = train_test_split(
        X_hg, y_hg, test_size=0.2, random_state=RANDOM_SEED, stratify=y_hg
    )

    # 域自适应权重
    print("\n🎯 域自适应权重...")
    domain_w = compute_domain_weights(X_hg_train, X_kz)
    print(f"    mean={domain_w.mean():.3f}, std={domain_w.std():.3f}")

    # ====== 阶段1：红光预训练 ======
    print(f"\n{'='*50}")
    print(f"🚀 阶段1: 红光(+合成)预训练")
    print(f"{'='*50}")

    n_pos = y_hg_train.sum(); n_neg = len(y_hg_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight = {spw:.2f}")

    params1 = XGBOOST_PRETRAIN.copy()
    es1 = params1.pop("early_stopping_rounds", 30)

    model_pretrain = xgb.XGBClassifier(
        scale_pos_weight=spw, use_label_encoder=False, **params1,
    )
    model_pretrain.fit(
        X_hg_train, y_hg_train, sample_weight=domain_w,
        eval_set=[(X_hg_val, y_hg_val)], verbose=50,
    )

    # 预训练 验证集评估
    pre_val = evaluate(model_pretrain, X_hg_val, y_hg_val, "预训练_验证集_红光", task_name)
    # 预训练 喀左测试集评估（作为 baseline）
    pre_kz = evaluate(model_pretrain, X_kz_test, y_kz_test, "预训练_喀左测试", task_name)

    # ====== 阶段2：喀左微调 ======
    print(f"\n{'='*50}")
    print(f"🔧 阶段2: 喀左 {FINETUNE_RATIO*100:.0f}% 微调")
    print(f"{'='*50}")

    # 合并：红光验证集 + 喀左微调集（少量）作为微调训练集
    # 策略：主要用喀左微调数据，辅以部分红光数据防止灾难性遗忘
    X_ft = pd.concat([X_hg_val, X_kz_ft], axis=0)
    y_ft = pd.concat([y_hg_val, y_kz_ft], axis=0)

    # 给喀左微调数据更高权重（5x）
    w_ft = np.ones(len(X_ft))
    w_ft[len(X_hg_val):] = 5.0  # 喀左样本 5x 权重
    print(f"  微调训练: {len(X_ft)} 样本 (红光验证={len(X_hg_val)}, 喀左={len(X_kz_ft)})")
    print(f"  喀左样本权重: 5.0x")

    n_pos_ft = y_ft.sum(); n_neg_ft = len(y_ft) - n_pos_ft
    spw_ft = n_neg_ft / max(n_pos_ft, 1)

    params2 = XGBOOST_FINETUNE.copy()
    es2 = params2.pop("early_stopping_rounds", 20)

    # 用预训练模型的 booster 初始化微调模型
    model_finetune = xgb.XGBClassifier(
        scale_pos_weight=spw_ft, use_label_encoder=False, **params2,
    )

    # 微调：使用 xgb_model 参数进行增量训练（迁移学习核心）
    model_finetune.fit(
        X_ft, y_ft, sample_weight=w_ft,
        eval_set=[(X_kz_test.head(500), y_kz_test.head(500))],  # 小量验证
        xgb_model=model_pretrain.get_booster(),  # 继承预训练权重
        verbose=50,
    )

    # 保存模型
    model_finetune.save_model(str(OUTPUT_DIR / f"xgb_{task_name}_v1.2.json"))

    # ====== 阈值优化（在喀左微调集上搜索） ======
    y_ft_prob = model_finetune.predict_proba(X_kz_ft)[:, 1]
    opt_thr, opt_f1 = find_optimal_threshold(y_kz_ft, y_ft_prob)
    print(f"\n🎯 最优阈值: {opt_thr:.2f} (微调集 F1={opt_f1:.4f})")

    # ====== 最终评估 ======
    val_m = evaluate(model_finetune, X_hg_val, y_hg_val, "验证集_红光", task_name, opt_thr)
    test_m = evaluate(model_finetune, X_kz_test, y_kz_test, "测试集_喀左_90%", task_name, opt_thr)

    # SHAP
    imp = shap_analysis(model_finetune, X_hg_val, list(feature_cols), task_name, "验证集")

    # ====== 三版对比 ======
    v10 = load_prev_report(task_name, "v1.0", V1_OUTPUT)
    v11 = load_prev_report(task_name, "v1.1", V11_OUTPUT)

    comparison = {"任务": task_name}

    versions = []
    if v10:
        versions.append(("v1.0", v10.get("独立测试集_喀左", {})))
    if v11:
        versions.append(("v1.1", v11.get("独立测试集_喀左", {})))
    versions.append(("v1.2", test_m))

    print(f"\n{'='*60}")
    print(f"📊 版本对比 — {task_name} (喀左测试集)")
    print(f"{'='*60}")
    for ver, data in versions:
        auc = data.get("ROC-AUC", "N/A")
        f1v = data.get("F1-Score", "N/A")
        rec = data.get("Recall", "N/A")
        auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        f1_s = f"{f1v:.4f}" if isinstance(f1v, float) else str(f1v)
        rec_s = f"{rec:.4f}" if isinstance(rec, float) else str(rec)
        print(f"  {ver}: AUC={auc_s}, F1={f1_s}, Recall={rec_s}")
        comparison[ver] = {"AUC": auc, "F1": f1v, "Recall": rec}

    # ====== 报告 ======
    report = {
        "版本": "v1.2",
        "任务": task_name,
        "描述": task_config["description"],
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "改进点": [
            f"PINN 反事实合成正样本",
            f"两阶段迁移学习: 红光预训练 → 喀左 {FINETUNE_RATIO*100:.0f}% 微调",
            "域自适应实例加权",
            "PINN 物理特征 + 时序趋势特征",
            f"F1-最优阈值: {opt_thr:.2f}",
        ],
        "迁移学习": {
            "喀左微调比例": FINETUNE_RATIO,
            "喀左微调样本数": int(len(X_kz_ft)),
            "喀左测试样本数": int(len(X_kz_test)),
            "预训练_喀左AUC": pre_kz.get("ROC-AUC"),
            "微调后_喀左AUC": test_m.get("ROC-AUC"),
        },
        "验证集": {k: v for k, v in val_m.items() if isinstance(v, (int, float, str))},
        "测试集_喀左": {k: v for k, v in test_m.items() if isinstance(v, (int, float, str))},
        "Top10特征": imp.head(10)["特征"].tolist(),
        "版本对比": comparison,
    }

    with open(OUTPUT_DIR / f"report_{task_name}_v1.2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    return model_finetune, report


# ============================================================
# 入口
# ============================================================
def main():
    print("=" * 70)
    print("  🐟🥬 病害预测 v1.2 — PINN 反事实合成 + 迁移学习")
    print("=" * 70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出: {OUTPUT_DIR}")
    print()

    # 加载数据（统一处理）
    all_reports = {}

    for task_name, task_config in TASKS.items():
        # 鱼类死亡才需要合成数据，蔬菜病害已经够好了
        use_syn = (task_name == "鱼类死亡")
        n_syn = 500 if use_syn else 0

        df_hg, df_kz, feature_cols = load_all_data(
            task_config, use_synthetic=use_syn, n_synthetic=n_syn
        )
        model, report = run_task(task_name, task_config, df_hg, df_kz, feature_cols)
        all_reports[task_name] = report

    # 汇总
    with open(OUTPUT_DIR / "summary_v1.2.json", "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n\n{'='*70}")
    print(f"  ✅ v1.2 完成!")
    print(f"{'='*70}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {p.name:55s} ({p.stat().st_size/1024:.1f} KB)")

    print(f"\n{'='*70}")
    print(f"  📊 版本演进总结 (喀左测试集)")
    print(f"{'='*70}")
    for tn, rep in all_reports.items():
        vc = rep.get("版本对比", {})
        print(f"\n  {tn}:")
        for ver in ["v1.0", "v1.1", "v1.2"]:
            d = vc.get(ver, {})
            if d:
                print(f"    {ver}: AUC={d.get('AUC','?')}, F1={d.get('F1','?')}")


if __name__ == "__main__":
    main()
