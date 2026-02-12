"""
XGBoost 训练与评估模块
- 训练 XGBoost 二分类模型
- 5折交叉验证
- 独立测试集评估（红光训练 → 喀左测试）
- SHAP 可解释性分析
- 输出报告与可视化
"""
import sys
import warnings
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.model_selection import StratifiedKFold

from config import XGBOOST_PARAMS, CV_FOLDS, SHAP_TOP_N, RANDOM_SEED, OUTPUT_DIR, TASKS
from data_loader import load_train_test

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 中文字体设置
# ============================================================
def setup_chinese_font():
    """配置 matplotlib 中文显示"""
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
    # fallback: 尝试系统默认
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("⚠️ 未找到理想中文字体，使用 fallback")

setup_chinese_font()


# ============================================================
# 训练
# ============================================================
def train_xgboost(X_train, y_train, X_val, y_val, task_name: str):
    """训练 XGBoost 模型"""
    # 计算类别权重（处理不平衡）
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"\n🔧 class imbalance — scale_pos_weight = {scale_pos_weight:.2f}")

    params = XGBOOST_PARAMS.copy()
    early_stop = params.pop("early_stopping_rounds", 30)

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        **params,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # 保存模型
    model_path = OUTPUT_DIR / f"xgb_{task_name}.json"
    model.save_model(str(model_path))
    print(f"💾 模型已保存: {model_path}")

    return model


# ============================================================
# 评估
# ============================================================
def evaluate(model, X, y, dataset_name: str, task_name: str):
    """评估模型性能并生成可视化"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # 基础指标
    metrics = {
        "数据集": dataset_name,
        "任务": task_name,
        "样本数": len(y),
        "正样本数": int(y.sum()),
        "正样本比例": f"{y.mean()*100:.1f}%",
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1-Score": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_prob) if y.nunique() > 1 else 0,
        "AP (Average Precision)": average_precision_score(y, y_prob) if y.nunique() > 1 else 0,
    }

    print(f"\n{'─'*50}")
    print(f"📈 评估结果 — {dataset_name} ({task_name})")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")

    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"  FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")

    # === 可视化 ===

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{task_name} — {dataset_name}", fontsize=14, fontweight="bold")

    # 1) 混淆矩阵热力图
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["无", "有"], yticklabels=["无", "有"])
    ax.set_xlabel("预测")
    ax.set_ylabel("实际")
    ax.set_title("混淆矩阵")

    # 2) ROC 曲线
    ax = axes[1]
    if y.nunique() > 1:
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {metrics['ROC-AUC']:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC 曲线")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "标签无变化\n无法绘制", ha="center", va="center")

    # 3) Precision-Recall 曲线
    ax = axes[2]
    if y.nunique() > 1:
        prec, rec, _ = precision_recall_curve(y, y_prob)
        ax.plot(rec, prec, "r-", lw=2, label=f"AP = {metrics['AP (Average Precision)']:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall 曲线")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "标签无变化\n无法绘制", ha="center", va="center")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"eval_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 评估图表已保存: {fig_path}")

    return metrics


# ============================================================
# 交叉验证
# ============================================================
def cross_validate(X, y, task_name: str):
    """5折交叉验证"""
    print(f"\n{'='*50}")
    print(f"🔄 {CV_FOLDS}折交叉验证 — {task_name}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = XGBOOST_PARAMS.copy()
    params.pop("early_stopping_rounds", None)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            **params,
        )
        model.fit(X_tr, y_tr, verbose=0)

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
def shap_analysis(model, X, feature_names, task_name: str, dataset_name: str = ""):
    """SHAP 可解释性分析"""
    print(f"\n{'='*50}")
    print(f"🔍 SHAP 分析 — {task_name} ({dataset_name})")
    print(f"{'='*50}")

    # 使用 TreeExplainer（XGBoost 专用，速度快）
    explainer = shap.TreeExplainer(model)

    # 如果数据量太大，抽样
    if len(X) > 5000:
        X_sample = X.sample(5000, random_state=RANDOM_SEED)
        print(f"  采样 5000 条进行 SHAP 分析（原始 {len(X)} 条）")
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # === 1. SHAP Summary Plot（蜂群图）===
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        max_display=SHAP_TOP_N,
        show=False,
    )
    plt.title(f"{task_name} — SHAP 特征重要性（Top {SHAP_TOP_N}）", fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"shap_summary_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 SHAP Summary Plot 已保存: {fig_path}")

    # === 2. SHAP Bar Plot（柱状图）===
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=SHAP_TOP_N,
        show=False,
    )
    plt.title(f"{task_name} — SHAP 平均绝对值（Top {SHAP_TOP_N}）", fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"shap_bar_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 SHAP Bar Plot 已保存: {fig_path}")

    # === 3. Top-3 特征的 Dependence Plot ===
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, idx in enumerate(top3_idx):
        shap.dependence_plot(
            idx, shap_values, X_sample,
            feature_names=feature_names,
            ax=axes[i],
            show=False,
        )
    fig.suptitle(f"{task_name} — Top 3 特征依赖图", fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"shap_dependence_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 SHAP Dependence Plot 已保存: {fig_path}")

    # === 4. 单样本 Waterfall Plot ===
    # 挑一个正样本展示归因
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=X_sample.values,
        feature_names=feature_names,
    )
    # 找一个正预测概率较高的样本
    probs = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argmax(probs)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(explanation[high_risk_idx], max_display=15, show=False)
    plt.title(f"{task_name} — 高风险样本归因分析", fontsize=14)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"shap_waterfall_{task_name}_{dataset_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 SHAP Waterfall Plot 已保存: {fig_path}")

    # === 5. 输出特征重要性排名 ===
    importance_df = pd.DataFrame({
        "特征": feature_names,
        "SHAP均值": mean_abs_shap,
    }).sort_values("SHAP均值", ascending=False).reset_index(drop=True)
    importance_df.index += 1
    importance_df.index.name = "排名"

    csv_path = OUTPUT_DIR / f"shap_importance_{task_name}_{dataset_name}.csv"
    importance_df.to_csv(csv_path)
    print(f"  📄 SHAP 特征排名已保存: {csv_path}")

    print(f"\n  📋 Top {SHAP_TOP_N} 特征:")
    print(importance_df.head(SHAP_TOP_N).to_string())

    return importance_df


# ============================================================
# 主流程
# ============================================================
def run_task(task_name: str, task_config: dict, use_augmented: bool = False):
    """运行一个完整的预测任务"""
    print(f"\n{'#'*60}")
    print(f"## 任务: {task_name} — {task_config['description']}")
    print(f"{'#'*60}")

    # 1. 加载数据
    X_train, X_val, y_train, y_val, X_test, y_test, feature_names = \
        load_train_test(task_config, use_augmented=use_augmented)

    # 2. 交叉验证（红光数据内部）
    X_all_hg = pd.concat([X_train, X_val], axis=0)
    y_all_hg = pd.concat([y_train, y_val], axis=0)
    cv_results = cross_validate(X_all_hg, y_all_hg, task_name)

    # 3. 训练最终模型
    print(f"\n🚀 训练最终模型...")
    model = train_xgboost(X_train, y_train, X_val, y_val, task_name)

    # 4. 评估
    val_metrics = evaluate(model, X_val, y_val, "验证集_红光", task_name)
    test_metrics = evaluate(model, X_test, y_test, "测试集_喀左", task_name)

    # 5. SHAP 分析（在验证集上）
    importance = shap_analysis(model, X_val, feature_names, task_name, "验证集")

    # 6. 汇总报告
    report = {
        "任务": task_name,
        "描述": task_config["description"],
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "数据增强": use_augmented,
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
    }

    report_path = OUTPUT_DIR / f"report_{task_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 报告已保存: {report_path}")

    return model, report


# ============================================================
# 入口
# ============================================================
def main():
    print("=" * 70)
    print("  🐟🥬 鱼菜共生病害预测系统 — XGBoost + SHAP")
    print("=" * 70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()

    all_reports = {}
    for task_name, task_config in TASKS.items():
        model, report = run_task(task_name, task_config, use_augmented=False)
        all_reports[task_name] = report

    # 保存汇总
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n\n{'='*70}")
    print(f"  ✅ 全部任务完成！")
    print(f"{'='*70}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  包含文件:")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:50s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
