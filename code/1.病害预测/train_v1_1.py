"""
病害预测 v1.1：融合 PINN 物理特征 + 时序趋势 + 域自适应
==========================================================
改进点：
  1. PINN 物理特征注入（DO 亏损、温度修正耗氧、氧气压力指数等）
  2. 时序趋势特征（温度/溶氧短期趋势、累积高温低氧天数）
  3. 域自适应实例加权（基于特征分布相似度）
  4. 更强正则化（降低跨基地过拟合）
  5. 阈值优化（F1-optimal threshold）

输出目录：output_v1.1/
"""
import sys
import warnings
import json
from datetime import datetime
from pathlib import Path

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
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR = SCRIPT_DIR / "output_v1.1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# v1.0 输出（用于对比）
V1_OUTPUT_DIR = SCRIPT_DIR / "output"

# PINN 学习到的物理参数（两站平均值，跨域不变假设）
PINN_PARAMS = {
    "K_La": (1.3497 + 1.3666) / 2,         # 复氧传质系数 (day^-1)
    "R_fish_base": (1.5687 + 1.4456) / 2,   # 鱼基础耗氧率 (mg/L/day)
    "alpha_T": (0.0341 + 0.0335) / 2,       # 耗氧温度系数
    "R_bio": (0.7769 + 0.7251) / 2,         # 微生物耗氧率 (mg/L/day)
    "P_photo_rate": (0.0678 + 0.0709) / 2,  # 光合产氧速率
    "T_ref": 25.0,                           # 参考温度
}

# 预测任务定义
TASKS = {
    "蔬菜病害": {
        "label_col": "蔬菜_病害次数",
        "description": "蔬菜是否发生病害（二分类）",
    },
    "鱼类死亡": {
        "label_col": "鱼_死亡数量",
        "description": "鱼类是否发生死亡（二分类）",
    },
}

# v1.1 更强正则化的 XGBoost 参数
XGBOOST_PARAMS_V11 = {
    "n_estimators": 500,
    "max_depth": 4,              # 6→4 降低过拟合
    "learning_rate": 0.03,       # 0.05→0.03 更小步长
    "subsample": 0.7,            # 0.8→0.7
    "colsample_bytree": 0.7,     # 0.8→0.7
    "min_child_weight": 10,      # 5→10 更强约束
    "gamma": 0.3,                # 0.1→0.3
    "reg_alpha": 0.5,            # 0.1→0.5 L1 正则
    "reg_lambda": 3.0,           # 1.0→3.0 L2 正则
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
}

# 域偏移风险高的特征（跨基地语义不一致）
DOMAIN_SHIFT_FEATURES = [
    "能耗km/h",                   # 不同基地设备体系不同，含义不同
    "种植床1液位上限距种植床表面距离cm",  # 设备差异
    "种植床2液位上限距种植床表面距离cm",
]


# ============================================================
# 中文字体设置
# ============================================================
def setup_chinese_font():
    font_candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for fp in font_candidates:
        try:
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            print(f"✅ 使用中文字体: {prop.get_name()}")
            return
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

setup_chinese_font()


# ============================================================
# 物理特征计算（基于 PINN 学到的参数）
# ============================================================
def do_saturation(T):
    """饱和溶氧浓度 (Benson & Krause, 1984 简化版)"""
    return 14.62 - 0.3898 * T + 0.006969 * T**2 - 5.897e-5 * T**3


def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 PINN 学到的物理方程，计算跨基地不变的物理特征。

    新增特征：
      - pinn_DO_deficit:    DO_sat(T) - DO_actual，溶氧亏损
      - pinn_R_fish_T:      温度修正后的鱼耗氧率
      - pinn_reaeration:    复氧速率 K_La * (DO_sat - DO)
      - pinn_P_photo:       光合产氧量
      - pinn_oxygen_stress: 氧气净消耗（正值=缺氧风险）
      - pinn_DO_margin:     距 2mg/L 警戒线的余量
      - pinn_DO_sat_ratio:  实际DO / 饱和DO（物理归一化）
    """
    p = PINN_PARAMS
    T_water = df["水温_日均"].fillna(25.0)
    DO_actual = df["溶氧mg/L"].fillna(df["溶氧mg/L"].median())
    light_h = df["光照时长h"].fillna(0.0)

    DO_sat = do_saturation(T_water)

    out = pd.DataFrame(index=df.index)

    # 1. 溶氧亏损（DO_sat - DO_actual）
    out["pinn_DO_deficit"] = DO_sat - DO_actual

    # 2. 温度修正后鱼耗氧率
    out["pinn_R_fish_T"] = p["R_fish_base"] * (1.0 + p["alpha_T"] * (T_water - p["T_ref"]))

    # 3. 复氧速率
    out["pinn_reaeration"] = p["K_La"] * (DO_sat - DO_actual)

    # 4. 光合产氧
    out["pinn_P_photo"] = p["P_photo_rate"] * light_h

    # 5. 氧气净压力指数（正 = 缺氧风险，负 = 安全）
    out["pinn_oxygen_stress"] = (
        out["pinn_R_fish_T"] + p["R_bio"]
        - out["pinn_reaeration"]
        - out["pinn_P_photo"]
    )

    # 6. 距警戒线余量
    out["pinn_DO_margin"] = DO_actual - 2.0

    # 7. 物理归一化 DO（无量纲，可跨基地比较）
    out["pinn_DO_sat_ratio"] = DO_actual / DO_sat.clip(lower=1.0)

    return out


# ============================================================
# 时序趋势特征
# ============================================================
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强时序趋势特征（补充超出原始特征工程的部分）。

    新增：
      - 水温短期趋势（当前 - 3日均值）
      - 溶氧短期变化幅度
      - 水气温交互项
      - 累积高温天数（7日内水温>28℃）
      - 累积低氧天数（7日内溶氧<5）
      - 温度波动加速度（3日std的变化率）
    """
    out = pd.DataFrame(index=df.index)

    T_water = df["水温_日均"].fillna(25.0)
    T_air = df["气温_日均"].fillna(20.0)
    DO = df["溶氧mg/L"].fillna(df["溶氧mg/L"].median())

    # 短期趋势
    roll3_water = df.get("水温_日均_roll3d_mean")
    if roll3_water is not None:
        out["trend_水温_3d"] = T_water - roll3_water.fillna(T_water)
    else:
        out["trend_水温_3d"] = 0.0

    # 溶氧变化幅度（用变化率的绝对值衡量波动）
    do_change = df.get("溶氧_变化率")
    if do_change is not None:
        out["DO_volatility"] = do_change.fillna(0).abs()
    else:
        out["DO_volatility"] = 0.0

    # 交互项：水温 × 气温（双高温耦合风险）
    out["水气温_交互"] = T_water * T_air / 100.0  # 归一化

    # 水温偏离适宜范围（18~28℃）
    out["水温_偏离适宜"] = np.where(
        T_water > 28, T_water - 28,
        np.where(T_water < 18, 18 - T_water, 0)
    )

    # 7日累积高温天数（水温 > 28℃）
    high_temp = (T_water > 28).astype(float)
    out["高温天数_7d"] = high_temp.rolling(7, min_periods=1).sum().values

    # 7日累积低氧天数（溶氧 < 5 mg/L）
    low_do_flag = (DO < 5).astype(float)
    out["低氧天数_7d"] = low_do_flag.rolling(7, min_periods=1).sum().values

    # 温度波动加速度
    roll3_std = df.get("水温_日均_roll3d_std")
    if roll3_std is not None:
        out["水温波动_加速度"] = roll3_std.fillna(0).diff().fillna(0)
    else:
        out["水温波动_加速度"] = 0.0

    return out


# ============================================================
# 域自适应：实例加权
# ============================================================
def compute_domain_weights(X_source: pd.DataFrame, X_target: pd.DataFrame,
                            method: str = "density_ratio") -> np.ndarray:
    """
    域自适应实例加权：上调与目标域（喀左）分布相似的源域（红光）样本权重。

    方法：基于特征距离的密度比估计（Kernel Mean Matching 简化版）。
    使用 PCA 降维后计算每个源域样本到目标域质心的马氏距离。

    Returns:
        weights: (n_source,) 每个训练样本的权重
    """
    from sklearn.decomposition import PCA

    # 选择数值列
    common_cols = [c for c in X_source.columns if c in X_target.columns]

    # 标准化
    scaler = StandardScaler()
    src_scaled = scaler.fit_transform(X_source[common_cols].fillna(0))
    tgt_scaled = scaler.transform(X_target[common_cols].fillna(0))

    # PCA 降到 10 维
    n_comp = min(10, src_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    src_pca = pca.fit_transform(src_scaled)
    tgt_pca = pca.transform(tgt_scaled)

    # 目标域质心
    tgt_centroid = tgt_pca.mean(axis=0)

    # 每个源域样本到目标域质心的欧氏距离
    distances = np.sqrt(((src_pca - tgt_centroid) ** 2).sum(axis=1))

    # 距离 → 权重（高斯核：距离越近权重越高）
    sigma = np.median(distances) + 1e-8
    weights = np.exp(-0.5 * (distances / sigma) ** 2)

    # 归一化使得均值为 1（不改变总样本权重）
    weights = weights / weights.mean()

    # 裁剪极端值
    weights = np.clip(weights, 0.1, 5.0)

    print(f"  🎯 域自适应权重: min={weights.min():.3f}, max={weights.max():.3f}, "
          f"mean={weights.mean():.3f}, std={weights.std():.3f}")

    return weights


# ============================================================
# 数据加载 + 特征增强
# ============================================================
def load_and_enhance(task_config: dict, remove_domain_shift: bool = True):
    """
    加载数据并增加物理 + 时序特征。

    Returns:
        X_train, X_val, y_train, y_val: 红光训练/验证
        X_test, y_test: 喀左独立测试
        feature_names: 特征列表
        domain_weights: 训练集域自适应权重
    """
    label_col = task_config["label_col"]

    # 加载原始 featured 数据
    df_hg = pd.read_csv(FEATURED_HONGGUANG, parse_dates=["日期"])
    df_kz = pd.read_csv(FEATURED_KAZUO, parse_dates=["日期"])
    print(f"[红光] 加载: {df_hg.shape[0]} 行, {df_hg.shape[1]} 列")
    print(f"[喀左] 加载: {df_kz.shape[0]} 行, {df_kz.shape[1]} 列")

    # ====== 增加物理特征 ======
    physics_hg = compute_physics_features(df_hg)
    physics_kz = compute_physics_features(df_kz)
    df_hg = pd.concat([df_hg, physics_hg], axis=1)
    df_kz = pd.concat([df_kz, physics_kz], axis=1)
    print(f"  ✅ 新增 {physics_hg.shape[1]} 个 PINN 物理特征")

    # ====== 增加时序趋势特征 ======
    temporal_hg = compute_temporal_features(df_hg)
    temporal_kz = compute_temporal_features(df_kz)
    df_hg = pd.concat([df_hg, temporal_hg], axis=1)
    df_kz = pd.concat([df_kz, temporal_kz], axis=1)
    print(f"  ✅ 新增 {temporal_hg.shape[1]} 个时序趋势特征")

    # ====== 排除非特征列 ======
    exclude = set(META_COLS + LABEL_COLS + CUMULATIVE_LABEL_COLS)
    if remove_domain_shift:
        exclude.update(DOMAIN_SHIFT_FEATURES)
        print(f"  ⚠️ 移除 {len(DOMAIN_SHIFT_FEATURES)} 个高域偏移特征: {DOMAIN_SHIFT_FEATURES}")

    feature_cols = [c for c in df_hg.columns if c not in exclude]
    X_hg = df_hg[feature_cols].copy()
    y_hg = (df_hg[label_col] > 0).astype(int)

    # 确保喀左有相同特征
    for c in feature_cols:
        if c not in df_kz.columns:
            df_kz[c] = 0
    X_kz = df_kz[feature_cols].copy()
    y_kz = (df_kz[label_col] > 0).astype(int)

    # 缺失值处理
    for X in [X_hg, X_kz]:
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(0)

    feature_names = list(X_hg.columns)

    print(f"  特征数: {len(feature_names)}")
    print(f"  红光正样本: {y_hg.sum()} ({y_hg.mean()*100:.1f}%)")
    print(f"  喀左正样本: {y_kz.sum()} ({y_kz.mean()*100:.1f}%)")

    # 拆分训练/验证
    X_train, X_val, y_train, y_val = train_test_split(
        X_hg, y_hg, test_size=0.2, random_state=RANDOM_SEED, stratify=y_hg
    )

    # ====== 域自适应权重 ======
    print("\n🎯 计算域自适应实例权重...")
    domain_weights = compute_domain_weights(X_train, X_kz)

    return X_train, X_val, y_train, y_val, X_kz, y_kz, feature_names, domain_weights


# ============================================================
# 训练
# ============================================================
def train_xgboost_v11(X_train, y_train, X_val, y_val, sample_weight,
                       task_name: str):
    """训练带域自适应权重的 XGBoost"""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"\n🔧 class imbalance — scale_pos_weight = {scale_pos_weight:.2f}")

    params = XGBOOST_PARAMS_V11.copy()
    early_stop = params.pop("early_stopping_rounds", 50)

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        **params,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    model_path = OUTPUT_DIR / f"xgb_{task_name}_v1.1.json"
    model.save_model(str(model_path))
    print(f"💾 模型已保存: {model_path}")

    return model


# ============================================================
# 阈值优化
# ============================================================
def find_optimal_threshold(y_true, y_prob):
    """基于 F1 最大化的阈值搜索"""
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


# ============================================================
# 评估
# ============================================================
def evaluate(model, X, y, dataset_name: str, task_name: str, threshold=None):
    """评估模型性能，支持自定义阈值"""
    y_prob = model.predict_proba(X)[:, 1]

    if threshold is None:
        y_pred = model.predict(X)
        used_threshold = 0.5
    else:
        y_pred = (y_prob >= threshold).astype(int)
        used_threshold = threshold

    metrics = {
        "数据集": dataset_name,
        "任务": task_name,
        "阈值": used_threshold,
        "样本数": len(y),
        "正样本数": int(y.sum()),
        "正样本比例": f"{y.mean()*100:.1f}%",
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1-Score": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else 0,
        "AP": average_precision_score(y, y_prob) if y.nunique() > 1 else 0,
    }

    print(f"\n{'─'*55}")
    print(f"📈 评估: {dataset_name} ({task_name}) | 阈值={used_threshold:.2f}")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    cm = confusion_matrix(y, y_pred)
    print(f"\n  混淆矩阵:")
    print(f"  TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"  FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")

    # === 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"[v1.1] {task_name} — {dataset_name}", fontsize=14, fontweight="bold")

    # 混淆矩阵
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["无", "有"], yticklabels=["无", "有"])
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    ax.set_title("混淆矩阵")

    # ROC
    ax = axes[1]
    if y.nunique() > 1:
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {metrics['ROC-AUC']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC 曲线")
        ax.legend()

    # PR
    ax = axes[2]
    if y.nunique() > 1:
        prec, rec, _ = precision_recall_curve(y, y_prob)
        ax.plot(rec, prec, "r-", lw=2, label=f"AP = {metrics['AP']:.4f}")
        ax.axhline(y=y.mean(), color="gray", ls="--", alpha=0.5, label="随机基线")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR 曲线")
        ax.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"eval_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 图表: {fig_path}")

    return metrics


# ============================================================
# 交叉验证
# ============================================================
def cross_validate(X, y, sample_weight, task_name):
    """5折交叉验证"""
    print(f"\n{'='*50}")
    print(f"🔄 {CV_FOLDS}折交叉验证 — {task_name} (v1.1)")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = XGBOOST_PARAMS_V11.copy()
    params.pop("early_stopping_rounds", None)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        w_tr = sample_weight[train_idx] if sample_weight is not None else None

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            **params,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=0)

        y_pred = model.predict(X_vl)
        y_prob = model.predict_proba(X_vl)[:, 1]

        fold_metrics.append({
            "Fold": fold_i,
            "F1": f1_score(y_vl, y_pred, zero_division=0),
            "Precision": precision_score(y_vl, y_pred, zero_division=0),
            "Recall": recall_score(y_vl, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_vl, y_prob) if y_vl.nunique() > 1 else 0,
        })
        print(f"  Fold {fold_i}: F1={fold_metrics[-1]['F1']:.4f}, "
              f"AUC={fold_metrics[-1]['AUC']:.4f}")

    df_cv = pd.DataFrame(fold_metrics)
    print(f"\n  平均 F1:  {df_cv['F1'].mean():.4f} ± {df_cv['F1'].std():.4f}")
    print(f"  平均 AUC: {df_cv['AUC'].mean():.4f} ± {df_cv['AUC'].std():.4f}")
    return df_cv


# ============================================================
# SHAP 分析
# ============================================================
def shap_analysis(model, X, feature_names, task_name, dataset_name=""):
    """SHAP 可解释性分析"""
    print(f"\n{'='*50}")
    print(f"🔍 SHAP 分析 — {task_name} ({dataset_name}) [v1.1]")
    print(f"{'='*50}")

    explainer = shap.TreeExplainer(model)

    if len(X) > 5000:
        X_sample = X.sample(5000, random_state=RANDOM_SEED)
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        max_display=SHAP_TOP_N, show=False,
    )
    plt.title(f"[v1.1] {task_name} — SHAP (Top {SHAP_TOP_N})", fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"shap_summary_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names,
        plot_type="bar", max_display=SHAP_TOP_N, show=False,
    )
    plt.title(f"[v1.1] {task_name} — SHAP 均值 (Top {SHAP_TOP_N})", fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"shap_bar_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "SHAP均值": mean_abs_shap,
    }).sort_values("SHAP均值", ascending=False).reset_index(drop=True)
    importance_df.index += 1
    importance_df.index.name = "排名"

    csv_path = OUTPUT_DIR / f"shap_importance_{task_name}_{dataset_name}.csv"
    importance_df.to_csv(csv_path)

    # 标记物理/时序特征
    pinn_feats = [f for f in importance_df["特征"] if f.startswith("pinn_")]
    temporal_feats = [f for f in importance_df["特征"]
                      if f.startswith("trend_") or f in ["DO_volatility", "水气温_交互",
                         "水温_偏离适宜", "高温天数_7d", "低氧天数_7d", "水温波动_加速度"]]

    pinn_ranks = importance_df[importance_df["特征"].isin(pinn_feats)]
    temporal_ranks = importance_df[importance_df["特征"].isin(temporal_feats)]

    print(f"\n  📋 Top {SHAP_TOP_N} 特征:")
    print(importance_df.head(SHAP_TOP_N).to_string())
    print(f"\n  🔬 PINN 物理特征排名:")
    print(pinn_ranks.to_string() if len(pinn_ranks) > 0 else "    (无)")
    print(f"\n  📈 时序趋势特征排名:")
    print(temporal_ranks.to_string() if len(temporal_ranks) > 0 else "    (无)")

    return importance_df


# ============================================================
# v1.0 vs v1.1 对比
# ============================================================
def load_v1_report(task_name):
    """加载 v1.0 的结果用于对比"""
    report_path = V1_OUTPUT_DIR / f"report_{task_name}.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_comparison(task_name, v11_metrics_val, v11_metrics_test):
    """生成 v1.0 vs v1.1 对比报告"""
    v1 = load_v1_report(task_name)
    if v1 is None:
        return None

    comparison = {
        "任务": task_name,
        "验证集_红光": {
            "v1.0_AUC": v1.get("验证集", {}).get("ROC-AUC"),
            "v1.1_AUC": v11_metrics_val["ROC-AUC"],
            "v1.0_F1": v1.get("验证集", {}).get("F1-Score"),
            "v1.1_F1": v11_metrics_val["F1-Score"],
            "AUC_变化": f"{(v11_metrics_val['ROC-AUC'] - v1['验证集']['ROC-AUC'])*100:+.2f}%",
            "F1_变化": f"{(v11_metrics_val['F1-Score'] - v1['验证集']['F1-Score'])*100:+.2f}%",
        },
        "测试集_喀左": {
            "v1.0_AUC": v1.get("独立测试集_喀左", {}).get("ROC-AUC"),
            "v1.1_AUC": v11_metrics_test["ROC-AUC"],
            "v1.0_F1": v1.get("独立测试集_喀左", {}).get("F1-Score"),
            "v1.1_F1": v11_metrics_test["F1-Score"],
            "AUC_变化": f"{(v11_metrics_test['ROC-AUC'] - v1['独立测试集_喀左']['ROC-AUC'])*100:+.2f}%",
            "F1_变化": f"{(v11_metrics_test['F1-Score'] - v1['独立测试集_喀左']['F1-Score'])*100:+.2f}%",
        },
    }

    print(f"\n{'='*60}")
    print(f"📊 v1.0 vs v1.1 对比 — {task_name}")
    print(f"{'='*60}")
    for split_name, data in [("验证集_红光", comparison["验证集_红光"]),
                              ("测试集_喀左", comparison["测试集_喀左"])]:
        print(f"\n  {split_name}:")
        print(f"    AUC: {data['v1.0_AUC']:.4f} → {data['v1.1_AUC']:.4f} ({data['AUC_变化']})")
        print(f"    F1:  {data['v1.0_F1']:.4f} → {data['v1.1_F1']:.4f} ({data['F1_变化']})")

    return comparison


# ============================================================
# 主流程
# ============================================================
def run_task(task_name: str, task_config: dict):
    """运行 v1.1 预测任务"""
    print(f"\n{'#'*60}")
    print(f"## [v1.1] {task_name} — {task_config['description']}")
    print(f"{'#'*60}")

    # 1. 加载数据 + 特征增强
    X_train, X_val, y_train, y_val, X_test, y_test, feature_names, domain_weights = \
        load_and_enhance(task_config, remove_domain_shift=True)

    # 2. 交叉验证
    X_all_hg = pd.concat([X_train, X_val], axis=0)
    y_all_hg = pd.concat([y_train, y_val], axis=0)
    # 合并权重
    all_weights = np.concatenate([
        domain_weights,
        compute_domain_weights(X_val, X_test)
    ])
    cv_results = cross_validate(X_all_hg, y_all_hg, all_weights, task_name)

    # 3. 训练最终模型
    print(f"\n🚀 训练 v1.1 模型...")
    model = train_xgboost_v11(X_train, y_train, X_val, y_val,
                               domain_weights, task_name)

    # 4. 在验证集上找最优阈值
    y_val_prob = model.predict_proba(X_val)[:, 1]
    opt_thr, opt_f1 = find_optimal_threshold(y_val, y_val_prob)
    print(f"\n🎯 最优阈值: {opt_thr:.2f} (验证集 F1={opt_f1:.4f})")

    # 5. 评估
    val_metrics = evaluate(model, X_val, y_val, "验证集_红光", task_name, threshold=opt_thr)
    test_metrics = evaluate(model, X_test, y_test, "测试集_喀左", task_name, threshold=opt_thr)

    # 6. SHAP 分析
    importance = shap_analysis(model, X_val, feature_names, task_name, "验证集")

    # 7. v1.0 vs v1.1 对比
    comparison = generate_comparison(task_name, val_metrics, test_metrics)

    # 8. 汇总报告
    report = {
        "版本": "v1.1",
        "任务": task_name,
        "描述": task_config["description"],
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "改进点": [
            "PINN 物理特征注入（7个新特征）",
            "时序趋势特征增强（7个新特征）",
            "域自适应实例加权",
            "移除高域偏移特征",
            "更强正则化（max_depth=4, L1/L2增强）",
            f"F1-最优阈值: {opt_thr:.2f}",
        ],
        "PINN参数": PINN_PARAMS,
        "交叉验证": {
            "F1_mean": float(cv_results["F1"].mean()),
            "F1_std": float(cv_results["F1"].std()),
            "AUC_mean": float(cv_results["AUC"].mean()),
            "AUC_std": float(cv_results["AUC"].std()),
        },
        "验证集": {k: v for k, v in val_metrics.items()
                    if isinstance(v, (int, float, str))},
        "独立测试集_喀左": {k: v for k, v in test_metrics.items()
                           if isinstance(v, (int, float, str))},
        "Top10特征": importance.head(10)["特征"].tolist(),
        "vs_v1.0": comparison,
    }

    report_path = OUTPUT_DIR / f"report_{task_name}_v1.1.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 报告: {report_path}")

    return model, report


def main():
    print("=" * 70)
    print("  🐟🥬 鱼菜共生病害预测 v1.1 — PINN + 时序 + 域自适应")
    print("=" * 70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  改进版本: v1.1")
    print()

    all_reports = {}
    all_comparisons = {}

    for task_name, task_config in TASKS.items():
        model, report = run_task(task_name, task_config)
        all_reports[task_name] = report
        if report.get("vs_v1.0"):
            all_comparisons[task_name] = report["vs_v1.0"]

    # 保存汇总
    summary_path = OUTPUT_DIR / "summary_v1.1.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, default=str)

    comparison_path = OUTPUT_DIR / "comparison_v1_vs_v1.1.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n\n{'='*70}")
    print(f"  ✅ v1.1 全部完成！")
    print(f"{'='*70}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  包含文件:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:55s} ({size_kb:.1f} KB)")

    # 最终总结
    print(f"\n{'='*70}")
    print(f"  📊 v1.0 → v1.1 改进总结")
    print(f"{'='*70}")
    for task_name, comp in all_comparisons.items():
        print(f"\n  {task_name}:")
        print(f"    验证集 AUC: {comp['验证集_红光']['AUC_变化']}")
        print(f"    测试集 AUC: {comp['测试集_喀左']['AUC_变化']}")
        print(f"    测试集 F1:  {comp['测试集_喀左']['F1_变化']}")


if __name__ == "__main__":
    main()
