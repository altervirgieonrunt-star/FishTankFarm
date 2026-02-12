"""
Chronos-T5-Tiny 时序预测脚本
- 加载本地 chronos-t5-tiny 模型
- 对水温、溶氧、氨氮等关键环境参数进行 Zero-Shot 预测
- 评估 MAE / RMSE / 置信区间覆盖率
- 输出预测 vs 真实对比图
"""
import sys
import re
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
# 路径配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = SCRIPT_DIR / "models" / "chronos-t5-tiny"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """将列名中的特殊字符替换为下划线，生成安全文件名"""
    return re.sub(r'[/\\:*?"<>|]', '_', name)

# ============================================================
# 预测目标配置
# ============================================================
TARGET_COLS = {
    "水温_日均": {"unit": "℃", "description": "日均水温"},
    "溶氧mg/L": {"unit": "mg/L", "description": "溶氧浓度"},
    "氨氮mg/L": {"unit": "mg/L", "description": "氨氮浓度"},
    "气温_日均": {"unit": "℃", "description": "日均气温"},
    "PH": {"unit": "", "description": "pH值"},
}

# 预测参数
CONTEXT_LENGTH = 128      # 用多少天历史做输入
PREDICTION_LENGTH = 14    # 预测未来多少天
NUM_SAMPLES = 50          # 概率预测采样数

# ============================================================
# 中文字体
# ============================================================
def setup_chinese_font():
    font_candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for fp in font_candidates:
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
# 数据加载
# ============================================================
def load_data(site: str = "红光"):
    """加载清洗后的数据，按模块分组取均值得到每日一条记录"""
    path = DATA_DIR / f"cleaned_{site}.csv"
    df = pd.read_csv(path, parse_dates=["日期"])
    
    # 按日期聚合（多模块取均值）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    daily = df.groupby("日期")[numeric_cols].mean().sort_index()
    
    print(f"📊 [{site}] 加载 {path.name}: 原始 {len(df)} 行 → 按日聚合 {len(daily)} 天")
    print(f"   日期范围: {daily.index.min()} ~ {daily.index.max()}")
    
    return daily


# ============================================================
# 单变量预测
# ============================================================
def predict_single(pipeline, series: np.ndarray, col_name: str, col_info: dict):
    """
    对单个时序列进行预测
    
    Returns:
        dict with predictions, actuals, metrics
    """
    # 拆分：最后 PREDICTION_LENGTH 天作为真实值
    if len(series) < CONTEXT_LENGTH + PREDICTION_LENGTH:
        print(f"  ⚠️ {col_name}: 有效数据不足 ({len(series)} < {CONTEXT_LENGTH + PREDICTION_LENGTH})，跳过")
        return None

    context = series[-(CONTEXT_LENGTH + PREDICTION_LENGTH):-PREDICTION_LENGTH]
    actual = series[-PREDICTION_LENGTH:]

    # 转为 torch tensor
    context_tensor = torch.tensor(context, dtype=torch.float32)

    # 预测
    forecast = pipeline.predict(
        context_tensor,
        prediction_length=PREDICTION_LENGTH,
        num_samples=NUM_SAMPLES,
    )
    # forecast shape: (1, num_samples, prediction_length)
    forecast_np = forecast.numpy().squeeze(0)  # (num_samples, prediction_length)

    # 统计
    median = np.median(forecast_np, axis=0)
    mean = np.mean(forecast_np, axis=0)
    low = np.percentile(forecast_np, 10, axis=0)
    high = np.percentile(forecast_np, 90, axis=0)

    # 评估指标
    mae = np.mean(np.abs(actual - median))
    rmse = np.sqrt(np.mean((actual - median) ** 2))
    # 置信区间覆盖率：实际值落在 [10%, 90%] 区间的比例
    coverage = np.mean((actual >= low) & (actual <= high))
    # MAPE
    mape = np.mean(np.abs((actual - median) / (actual + 1e-8))) * 100

    result = {
        "col_name": col_name,
        "unit": col_info["unit"],
        "description": col_info["description"],
        "context": context,
        "actual": actual,
        "median": median,
        "mean": mean,
        "low_10": low,
        "high_90": high,
        "all_samples": forecast_np,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Coverage_80": coverage,
    }

    print(f"  ✅ {col_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, "
          f"MAPE={mape:.1f}%, Coverage(80%)={coverage:.1%}")

    return result


# ============================================================
# 可视化
# ============================================================
def plot_forecast(result: dict, site: str):
    """绘制单变量预测图"""
    fig, ax = plt.subplots(figsize=(14, 5))

    n_ctx = len(result["context"])
    n_pred = len(result["actual"])

    # 历史数据
    ctx_x = range(n_ctx)
    ax.plot(ctx_x, result["context"], "b-", alpha=0.5, linewidth=1, label="历史数据")

    # 预测区间
    pred_x = range(n_ctx, n_ctx + n_pred)
    ax.fill_between(pred_x, result["low_10"], result["high_90"],
                    alpha=0.2, color="orange", label="80%置信区间")
    ax.plot(pred_x, result["median"], "r-", linewidth=2, label="预测中位数")
    ax.plot(pred_x, result["actual"], "g--", linewidth=2, marker="o",
            markersize=4, label="真实值")

    # 分隔线
    ax.axvline(x=n_ctx - 0.5, color="gray", linestyle="--", alpha=0.5)

    col = result["col_name"]
    unit = result["unit"]
    ax.set_title(
        f"{site} — {result['description']}（{col}）预测\n"
        f"MAE={result['MAE']:.4f}{unit}  RMSE={result['RMSE']:.4f}{unit}  "
        f"MAPE={result['MAPE']:.1f}%  Coverage(80%)={result['Coverage_80']:.0%}",
        fontsize=12
    )
    ax.set_xlabel(f"天 (最后{n_ctx}天历史 → 预测{n_pred}天)")
    ax.set_ylabel(f"{col} ({unit})" if unit else col)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"forecast_{safe_filename(col)}_{site}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


def plot_all_summary(all_results: dict, site: str):
    """所有变量汇总图"""
    valid = {k: v for k, v in all_results.items() if v is not None}
    n = len(valid)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (col_name, r) in zip(axes, valid.items()):
        n_ctx = len(r["context"])
        n_pred = len(r["actual"])
        ctx_x = range(n_ctx)
        pred_x = range(n_ctx, n_ctx + n_pred)

        ax.plot(ctx_x, r["context"], "b-", alpha=0.4, linewidth=1)
        ax.fill_between(pred_x, r["low_10"], r["high_90"],
                        alpha=0.2, color="orange")
        ax.plot(pred_x, r["median"], "r-", linewidth=2, label="预测")
        ax.plot(pred_x, r["actual"], "g--", linewidth=2, marker="o",
                markersize=3, label="真实")
        ax.axvline(x=n_ctx - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{r['description']}\n({r['unit']})" if r['unit'] else r['description'])
        ax.set_title(f"{col_name}  |  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                     f"MAPE={r['MAPE']:.1f}%  Cov={r['Coverage_80']:.0%}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Chronos-T5-Tiny Zero-Shot 预测汇总 — {site}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"forecast_summary_{site}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    return fig_path


# ============================================================
# 主流程
# ============================================================
def run(site: str = "红光"):
    """运行完整预测流程"""
    import json
    from chronos import ChronosPipeline

    print(f"\n{'='*60}")
    print(f"  🔮 Chronos-T5-Tiny 时序预测 — {site}")
    print(f"{'='*60}")
    print(f"  模型: {MODEL_DIR}")
    print(f"  上下文窗口: {CONTEXT_LENGTH} 天")
    print(f"  预测长度: {PREDICTION_LENGTH} 天")
    print(f"  采样数: {NUM_SAMPLES}")
    print()

    # 1. 加载模型
    print("🔧 加载 Chronos 模型...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   设备: {device}")

    pipeline = ChronosPipeline.from_pretrained(
        str(MODEL_DIR),
        device_map=device,
        dtype=torch.float32,
    )
    print("   ✅ 模型加载完成\n")

    # 2. 加载数据
    daily = load_data(site)

    # 3. 逐变量预测
    all_results = {}
    metrics_list = []

    for col_name, col_info in TARGET_COLS.items():
        print(f"\n📈 预测: {col_name} ({col_info['description']})")

        if col_name not in daily.columns:
            print(f"  ⚠️ 列 {col_name} 不存在，跳过")
            continue

        series = daily[col_name].dropna().values
        print(f"  有效数据: {len(series)} 天")

        result = predict_single(pipeline, series, col_name, col_info)
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
            })

    # 4. 汇总图
    summary_fig = plot_all_summary(all_results, site)
    if summary_fig:
        print(f"\n📊 汇总图已保存: {summary_fig}")

    # 5. 指标汇总表
    if metrics_list:
        df_metrics = pd.DataFrame(metrics_list)
        print(f"\n{'='*60}")
        print(f"📊 评估指标汇总 — {site}")
        print(f"{'='*60}")
        print(df_metrics.to_string(index=False))

        csv_path = OUTPUT_DIR / f"metrics_{site}.csv"
        df_metrics.to_csv(csv_path, index=False)

        # 保存 JSON 报告
        report = {
            "模型": "chronos-t5-tiny",
            "站点": site,
            "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "上下文窗口": CONTEXT_LENGTH,
            "预测长度": PREDICTION_LENGTH,
            "采样数": NUM_SAMPLES,
            "指标": metrics_list,
        }
        report_path = OUTPUT_DIR / f"report_{site}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n📄 报告已保存: {report_path}")

    return all_results


def main():
    print("=" * 60)
    print("  🔮 Chronos-T5-Tiny Zero-Shot 时序预测")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出目录: {OUTPUT_DIR}")

    # 红光和喀左都跑
    for site in ["红光", "喀左"]:
        run(site)

    print(f"\n\n{'='*60}")
    print(f"  ✅ 全部站点预测完成！")
    print(f"{'='*60}")
    print(f"  输出目录: {OUTPUT_DIR}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:50s} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
