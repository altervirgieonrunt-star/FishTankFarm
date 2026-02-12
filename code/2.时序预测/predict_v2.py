"""
Chronos-T5-Tiny 时序预测 v2 — 对策改进版
改进点：
  1. context 窗口缩短至 48 天 → 氨氮/pH 可预测
  2. 使用模块级数据（不按日聚合）→ 更多数据、更细粒度
  3. 多模块独立预测 + 集成 → 更鲁棒的预测
"""
import sys
import re
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

# ============================================================
# 路径
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = SCRIPT_DIR / "models" / "chronos-t5-tiny"
OUTPUT_DIR = SCRIPT_DIR / "output2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', '_', name)


# ============================================================
# 改进参数
# ============================================================
TARGET_COLS = {
    "水温_日均": {"unit": "℃", "description": "日均水温"},
    "溶氧mg/L": {"unit": "mg/L", "description": "溶氧浓度"},
    "氨氮mg/L": {"unit": "mg/L", "description": "氨氮浓度"},
    "气温_日均": {"unit": "℃", "description": "日均气温"},
    "PH": {"unit": "", "description": "pH值"},
}

CONTEXT_LENGTH = 48       # ← 缩短到 48 天（原128）
PREDICTION_LENGTH = 14
NUM_SAMPLES = 50
MAX_MODULES = 5           # 每个变量最多使用前N个数据最多的模块


# ============================================================
# 中文字体
# ============================================================
def setup_chinese_font():
    for fp in ["/System/Library/Fonts/STHeiti Light.ttc",
               "/System/Library/Fonts/PingFang.ttc",
               "/System/Library/Fonts/Supplemental/Songti.ttc"]:
        try:
            fm.fontManager.addfont(fp)
            prop = fm.FontProperties(fname=fp)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

setup_chinese_font()


# ============================================================
# 加载模块级数据
# ============================================================
def load_module_data(site: str):
    """加载清洗后数据，返回按 (模块, 日期) 排序的 DataFrame"""
    path = DATA_DIR / f"cleaned_{site}.csv"
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df.sort_values(["模块", "日期"]).reset_index(drop=True)
    modules = df["模块"].nunique()
    print(f"📊 [{site}] 加载: {len(df)} 行, {modules} 个模块")
    return df


def get_best_modules(df, col_name, n=MAX_MODULES):
    """找出某个变量有效数据最多的 top-N 模块"""
    valid = df.dropna(subset=[col_name])
    counts = valid.groupby("模块").size().sort_values(ascending=False)
    # 只选数据量 >= context + prediction 的模块
    min_rows = CONTEXT_LENGTH + PREDICTION_LENGTH
    eligible = counts[counts >= min_rows]
    selected = eligible.head(n).index.tolist()
    return selected, counts


# ============================================================
# 单模块预测
# ============================================================
def predict_module(pipeline, series: np.ndarray):
    """对一个模块的时序进行预测，返回 forecast array"""
    min_len = CONTEXT_LENGTH + PREDICTION_LENGTH
    if len(series) < min_len:
        return None, None

    context = series[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    actual = series[-PREDICTION_LENGTH:]
    context_tensor = torch.tensor(context, dtype=torch.float32)

    forecast = pipeline.predict(
        context_tensor,
        prediction_length=PREDICTION_LENGTH,
        num_samples=NUM_SAMPLES,
    )
    forecast_np = forecast.numpy().squeeze(0)  # (num_samples, pred_len)
    return forecast_np, actual


def compute_metrics(actual, median, low, high):
    """计算评估指标"""
    mae = np.mean(np.abs(actual - median))
    rmse = np.sqrt(np.mean((actual - median) ** 2))
    mape = np.mean(np.abs((actual - median) / (np.abs(actual) + 1e-8))) * 100
    coverage = np.mean((actual >= low) & (actual <= high))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Coverage_80": coverage}


# ============================================================
# 多模块集成预测
# ============================================================
def ensemble_predict(pipeline, df, col_name, col_info, site):
    """
    对多个模块独立预测后集成：
    - 每个模块独立得到 forecast 分布
    - 将所有模块的 forecast samples 合并
    - 取合并分布的中位数作为最终预测
    """
    selected_modules, all_counts = get_best_modules(df, col_name)

    if len(selected_modules) == 0:
        # 回退：尝试日聚合
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        daily = df.groupby("日期")[numeric].mean().sort_index()
        series = daily[col_name].dropna().values
        if len(series) >= CONTEXT_LENGTH + PREDICTION_LENGTH:
            print(f"  💡 无模块满足要求，回退到日聚合数据 ({len(series)} 天)")
            forecast_np, actual = predict_module(pipeline, series)
            if forecast_np is not None:
                median = np.median(forecast_np, axis=0)
                low = np.percentile(forecast_np, 10, axis=0)
                high = np.percentile(forecast_np, 90, axis=0)
                metrics = compute_metrics(actual, median, low, high)
                return {
                    "col_name": col_name, "unit": col_info["unit"],
                    "description": col_info["description"],
                    "actual": actual, "median": median,
                    "low_10": low, "high_90": high,
                    "n_modules": 0, "method": "日聚合回退",
                    **metrics,
                }
        print(f"  ❌ {col_name}: 数据不足，无法预测")
        return None

    print(f"  📋 选取 {len(selected_modules)} 个模块: {selected_modules}")
    print(f"     各模块数据量: {[int(all_counts[m]) for m in selected_modules]}")

    all_forecasts = []
    actuals = []
    module_metrics = []

    for mod in selected_modules:
        mod_data = df[df["模块"] == mod].sort_values("日期")
        series = mod_data[col_name].dropna().values

        forecast_np, actual = predict_module(pipeline, series)
        if forecast_np is None:
            continue

        all_forecasts.append(forecast_np)
        actuals.append(actual)

        # 单模块指标
        med = np.median(forecast_np, axis=0)
        lo = np.percentile(forecast_np, 10, axis=0)
        hi = np.percentile(forecast_np, 90, axis=0)
        m = compute_metrics(actual, med, lo, hi)
        module_metrics.append({"module": mod, **m})
        print(f"    {mod}: MAE={m['MAE']:.4f}, MAPE={m['MAPE']:.1f}%")

    if not all_forecasts:
        print(f"  ❌ {col_name}: 所有模块预测失败")
        return None

    # 集成：合并所有模块的 forecast samples
    # 注意：不同模块的 actual 可能不同（不同时间段），
    # 所以我们用第一个模块的 actual 作为参考（时间最近的数据）
    ensemble_forecast = np.concatenate(all_forecasts, axis=0)
    ref_actual = actuals[0]  # 使用数据最多的模块的 actual

    ensemble_median = np.median(ensemble_forecast, axis=0)
    ensemble_low = np.percentile(ensemble_forecast, 10, axis=0)
    ensemble_high = np.percentile(ensemble_forecast, 90, axis=0)
    ensemble_metrics = compute_metrics(ref_actual, ensemble_median, ensemble_low, ensemble_high)

    print(f"  🔗 集成结果 ({len(all_forecasts)} 模块): "
          f"MAE={ensemble_metrics['MAE']:.4f}, RMSE={ensemble_metrics['RMSE']:.4f}, "
          f"MAPE={ensemble_metrics['MAPE']:.1f}%, Cov={ensemble_metrics['Coverage_80']:.0%}")

    # 也提供近 context 天的历史数据用于绘图
    best_mod = selected_modules[0]
    best_series = df[df["模块"] == best_mod].sort_values("日期")[col_name].dropna().values
    context_for_plot = best_series[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]

    return {
        "col_name": col_name,
        "unit": col_info["unit"],
        "description": col_info["description"],
        "context": context_for_plot,
        "actual": ref_actual,
        "median": ensemble_median,
        "low_10": ensemble_low,
        "high_90": ensemble_high,
        "n_modules": len(all_forecasts),
        "method": f"多模块集成({len(all_forecasts)}个)",
        "module_metrics": module_metrics,
        **ensemble_metrics,
    }


# ============================================================
# 可视化
# ============================================================
def plot_forecast(result, site):
    fig, ax = plt.subplots(figsize=(14, 5))
    col = result["col_name"]
    unit = result["unit"]
    n_pred = len(result["actual"])

    if "context" in result and result["context"] is not None:
        n_ctx = len(result["context"])
        ax.plot(range(n_ctx), result["context"], "b-", alpha=0.5, lw=1, label="历史数据")
        pred_x = range(n_ctx, n_ctx + n_pred)
    else:
        n_ctx = 0
        pred_x = range(n_pred)

    ax.fill_between(pred_x, result["low_10"], result["high_90"],
                    alpha=0.2, color="orange", label="80%置信区间")
    ax.plot(pred_x, result["median"], "r-", lw=2, label="预测中位数")
    ax.plot(pred_x, result["actual"], "g--", lw=2, marker="o", ms=4, label="真实值")

    if n_ctx > 0:
        ax.axvline(x=n_ctx - 0.5, color="gray", ls="--", alpha=0.5)

    method = result.get("method", "")
    ax.set_title(
        f"{site} — {result['description']}（{col}）预测 [{method}]\n"
        f"MAE={result['MAE']:.4f}{unit}  RMSE={result['RMSE']:.4f}{unit}  "
        f"MAPE={result['MAPE']:.1f}%  Coverage(80%)={result['Coverage_80']:.0%}",
        fontsize=11
    )
    ax.set_xlabel(f"天 (context={CONTEXT_LENGTH}天 → 预测{PREDICTION_LENGTH}天)")
    ax.set_ylabel(f"{col} ({unit})" if unit else col)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"forecast_{safe_filename(col)}_{site}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_summary(all_results, site):
    valid = {k: v for k, v in all_results.items() if v is not None}
    n = len(valid)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, (col_name, r) in zip(axes, valid.items()):
        n_pred = len(r["actual"])
        if "context" in r and r["context"] is not None:
            n_ctx = len(r["context"])
            ax.plot(range(n_ctx), r["context"], "b-", alpha=0.4, lw=1)
            pred_x = range(n_ctx, n_ctx + n_pred)
            ax.axvline(x=n_ctx - 0.5, color="gray", ls="--", alpha=0.5)
        else:
            pred_x = range(n_pred)

        ax.fill_between(pred_x, r["low_10"], r["high_90"], alpha=0.2, color="orange")
        ax.plot(pred_x, r["median"], "r-", lw=2, label="预测")
        ax.plot(pred_x, r["actual"], "g--", lw=2, marker="o", ms=3, label="真实")
        ax.set_ylabel(f"{r['description']}\n({r['unit']})" if r['unit'] else r['description'])
        ax.set_title(f"{col_name} [{r.get('method','')}] | MAE={r['MAE']:.4f} "
                     f"RMSE={r['RMSE']:.4f} MAPE={r['MAPE']:.1f}% "
                     f"Cov={r['Coverage_80']:.0%}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Chronos-T5-Tiny v2 预测汇总 — {site} (context={CONTEXT_LENGTH}天, 多模块集成)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"forecast_summary_{site}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


# ============================================================
# 对比表: v1 vs v2
# ============================================================
def load_v1_metrics(site):
    """加载 v1 的指标用于对比"""
    v1_path = SCRIPT_DIR / "output" / f"report_{site}.json"
    if v1_path.exists():
        with open(v1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {m["变量"]: m for m in data.get("指标", [])}
    return {}


def print_comparison(v1_metrics, v2_results, site):
    """打印 v1 vs v2 对比表"""
    print(f"\n{'='*70}")
    print(f"📊 v1 vs v2 对比 — {site}")
    print(f"{'='*70}")
    print(f"{'变量':>12s} | {'v1 MAE':>8s} → {'v2 MAE':>8s} | {'v1 MAPE':>8s} → {'v2 MAPE':>8s} | {'改进':>6s}")
    print("-" * 70)

    comparison = []
    for col_name, v2 in v2_results.items():
        if v2 is None:
            continue
        v1 = v1_metrics.get(col_name, {})
        v1_mae = v1.get("MAE", None)
        v1_mape = v1.get("MAPE(%)", None)
        v2_mae = v2["MAE"]
        v2_mape = v2["MAPE"]

        if v1_mae is not None:
            improve = ((v1_mae - v2_mae) / v1_mae * 100)
            print(f"{col_name:>12s} | {v1_mae:8.4f} → {v2_mae:8.4f} | "
                  f"{v1_mape:7.1f}% → {v2_mape:7.1f}% | {improve:+5.1f}%")
        else:
            print(f"{col_name:>12s} | {'N/A':>8s} → {v2_mae:8.4f} | "
                  f"{'N/A':>8s} → {v2_mape:7.1f}% | {'NEW':>6s}")

        comparison.append({
            "变量": col_name,
            "v1_MAE": v1_mae,
            "v2_MAE": float(v2_mae),
            "v1_MAPE": v1_mape,
            "v2_MAPE": float(v2_mape),
            "v2_Coverage": float(v2["Coverage_80"]),
            "方法": v2.get("method", ""),
        })

    return comparison


# ============================================================
# 主流程
# ============================================================
def run(pipeline, site):
    print(f"\n{'='*60}")
    print(f"  🔮 Chronos v2 — {site} (context={CONTEXT_LENGTH}天, 多模块集成)")
    print(f"{'='*60}\n")

    df = load_module_data(site)

    all_results = {}
    metrics_list = []

    for col_name, col_info in TARGET_COLS.items():
        print(f"\n📈 预测: {col_name} ({col_info['description']})")
        result = ensemble_predict(pipeline, df, col_name, col_info, site)
        all_results[col_name] = result

        if result is not None:
            fig_path = plot_forecast(result, site)
            print(f"  📊 图表已保存: {fig_path}")
            metrics_list.append({
                "变量": col_name,
                "描述": col_info["description"],
                "MAE": float(result["MAE"]),
                "RMSE": float(result["RMSE"]),
                "MAPE(%)": float(result["MAPE"]),
                "Coverage(80%)": float(result["Coverage_80"]),
                "方法": result.get("method", ""),
                "模块数": result.get("n_modules", 0),
            })

    # 汇总图
    summary_fig = plot_summary(all_results, site)
    if summary_fig:
        print(f"\n📊 汇总图已保存: {summary_fig}")

    # v1 vs v2 对比
    v1_metrics = load_v1_metrics(site)
    comparison = print_comparison(v1_metrics, all_results, site)

    # 保存报告
    report = {
        "模型": "chronos-t5-tiny",
        "版本": "v2 (对策改进版)",
        "站点": site,
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "改进点": [
            f"context 窗口: 128→{CONTEXT_LENGTH} 天",
            "使用模块级数据（不聚合）",
            f"多模块集成（最多{MAX_MODULES}个模块）",
        ],
        "指标": metrics_list,
        "v1_vs_v2": comparison,
    }
    report_path = OUTPUT_DIR / f"report_{site}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    df_metrics = pd.DataFrame(metrics_list)
    csv_path = OUTPUT_DIR / f"metrics_{site}.csv"
    df_metrics.to_csv(csv_path, index=False)

    print(f"\n📄 报告已保存: {report_path}")

    return all_results, comparison


def main():
    from chronos import ChronosPipeline

    print("=" * 60)
    print("  🔮 Chronos-T5-Tiny v2 — 对策改进版")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  改进: context {CONTEXT_LENGTH}天 + 模块级数据 + 多模块集成")
    print(f"  输出: {OUTPUT_DIR}\n")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 加载模型... (设备: {device})")
    pipeline = ChronosPipeline.from_pretrained(
        str(MODEL_DIR), device_map=device, dtype=torch.float32,
    )
    print("   ✅ 模型加载完成\n")

    all_comparisons = {}
    for site in ["红光", "喀左"]:
        _, comp = run(pipeline, site)
        all_comparisons[site] = comp

    # 汇总对比保存
    with open(OUTPUT_DIR / "comparison_v1_vs_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*60}")
    print(f"  ✅ v2 全部完成！")
    print(f"{'='*60}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {p.name:50s} ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
