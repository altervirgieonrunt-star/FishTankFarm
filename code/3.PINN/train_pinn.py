"""
PINN 训练主脚本

流程:
  1. 加载红光数据（日均聚合）
  2. 构建训练/验证集
  3. 训练 PINN (Data Loss + Physics Loss + Boundary Loss)
  4. 同时学习网络权重和物理参数
  5. 评估 + 可视化 + 反事实推理
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, OUTPUT_DIR, INPUT_COLS, TARGET_COL, AUX_COLS,
    PHYSICS_PARAMS, PARAM_BOUNDS, NET_CONFIG, TRAIN_CONFIG, RANDOM_SEED,
)
from physics import (
    PhysicsParams, do_saturation,
    compute_ode_residual, compute_boundary_loss,
)
from model import PINN


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
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 数据加载
# ============================================================
def load_data(site: str = "红光"):
    """加载并聚合数据，返回有效的日均 DataFrame"""
    path = DATA_DIR / f"cleaned_{site}.csv"
    df = pd.read_csv(path, parse_dates=["日期"])
    
    # 日均聚合
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    daily = df.groupby("日期")[numeric_cols].mean().sort_index()
    daily = daily.reset_index()
    
    # 确定可用特征列
    avail_input = [c for c in INPUT_COLS if c in daily.columns]
    avail_aux = [c for c in AUX_COLS if c in daily.columns]
    all_input = avail_input + avail_aux
    
    # 仅保留目标列非空的行
    mask = daily[TARGET_COL].notna()
    for c in avail_input:
        mask &= daily[c].notna()
    
    daily_clean = daily[mask].copy().reset_index(drop=True)
    
    # 添加归一化时间列 (0~1)
    dates = pd.to_datetime(daily_clean["日期"])
    t_days = (dates - dates.min()).dt.days.values.astype(float)
    t_norm = t_days / max(t_days.max(), 1.0)
    daily_clean["t_norm"] = t_norm
    daily_clean["t_days"] = t_days
    
    print(f"📊 [{site}] 有效数据: {len(daily_clean)} 天")
    print(f"   输入特征: {all_input}")
    print(f"   目标: {TARGET_COL}")
    print(f"   日期: {dates.min().date()} ~ {dates.max().date()}")
    
    return daily_clean, all_input


def prepare_tensors(df, input_cols, device):
    """将 DataFrame 转为 PyTorch 张量"""
    # 特征: [t_norm, input_cols...]
    feat_cols = ["t_norm"] + input_cols
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=device)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.float32, device=device)
    t = torch.tensor(df["t_norm"].values, dtype=torch.float32, device=device)
    t.requires_grad_(True)
    
    # 特征归一化统计
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0).clamp(min=1e-6)
    
    return X, y, t, X_mean, X_std


# ============================================================
# 训练
# ============================================================
def train_pinn(site: str = "红光"):
    """完整训练流程"""
    cfg = TRAIN_CONFIG
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  🧬 PINN 训练 — 溶氧动力学 ({site})")
    print(f"{'='*60}")
    print(f"  设备: {device}")
    
    # 1. 加载数据
    df, input_cols = load_data(site)
    n_total = len(df)
    
    # 2. 划分训练/验证（时间顺序，后 20% 为验证）
    n_val = int(n_total * cfg["val_ratio"])
    n_train = n_total - n_val
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:].copy()
    print(f"  训练: {n_train} 天, 验证: {n_val} 天")
    
    # 3. 准备张量
    X_train, y_train, t_train, X_mean, X_std = prepare_tensors(df_train, input_cols, device)
    X_val, y_val, t_val, _, _ = prepare_tensors(df_val, input_cols, device)
    
    # 归一化
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    
    # 提取水温和光照列索引
    feat_cols = ["t_norm"] + input_cols
    idx_temp = feat_cols.index("水温_日均")
    idx_light = feat_cols.index("光照时长h") if "光照时长h" in feat_cols else None
    
    # 4. 初始化模型和物理参数
    n_features = X_train_norm.shape[1]
    net = PINN(n_features, NET_CONFIG["hidden_layers"], NET_CONFIG["activation"]).to(device)
    physics = PhysicsParams(PHYSICS_PARAMS, PARAM_BOUNDS).to(device)
    
    print(f"  网络参数: {sum(p.numel() for p in net.parameters()):,}")
    print(f"  物理参数: {sum(p.numel() for p in physics.parameters())}")
    
    # 5. 优化器
    optimizer = torch.optim.Adam([
        {"params": net.parameters(), "lr": cfg["lr_net"]},
        {"params": physics.parameters(), "lr": cfg["lr_physics"]},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step"], gamma=cfg["scheduler_gamma"]
    )
    
    # 6. 训练循环
    history = {"epoch": [], "loss": [], "loss_data": [], "loss_physics": [],
               "loss_boundary": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    
    lambda_phys = cfg["lambda_physics"]
    lambda_bnd = cfg["lambda_boundary"]
    
    print(f"\n🚀 开始训练 ({cfg['epochs']} epochs)...")
    print(f"   λ_physics={lambda_phys}, λ_boundary={lambda_bnd}")
    
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        physics.train()
        
        # --- 构造需要梯度的输入 ---
        # t_input 需要 requires_grad=True 以支持自动微分 dDO/dt
        # 归一化 t: 使用训练集的 mean/std
        t_raw = X_train[:, 0:1].detach().clone()   # 未归一化的 t_norm
        t_input = ((t_raw - X_mean[0]) / X_std[0]).requires_grad_(True)
        
        # 其他特征使用预归一化的值（不需要梯度）
        other_norm = X_train_norm[:, 1:].detach()
        X_input = torch.cat([t_input, other_norm], dim=1)
        
        # 前向
        DO_pred = net(X_input)
        
        # Data Loss
        loss_data = nn.functional.mse_loss(DO_pred, y_train)
        
        # Physics Loss (ODE 残差)
        T_water = X_train[:, idx_temp]  # 未归一化的水温
        light = X_train[:, idx_light] if idx_light is not None else torch.zeros_like(T_water)
        
        residual = compute_ode_residual(DO_pred, t_input, T_water, light, physics)
        loss_physics = torch.mean(residual ** 2)
        
        # Boundary Loss
        loss_boundary = compute_boundary_loss(DO_pred, T_water)
        
        # Total Loss
        loss = loss_data + lambda_phys * loss_physics + lambda_bnd * loss_boundary
        
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation
        net.eval()
        with torch.no_grad():
            DO_val_pred = net(X_val_norm)
            val_mae = torch.mean(torch.abs(DO_val_pred - y_val)).item()
        
        # 记录
        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["loss_data"].append(loss_data.item())
        history["loss_physics"].append(loss_physics.item())
        history["loss_boundary"].append(loss_boundary.item())
        history["val_mae"].append(val_mae)
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                "net": {k: v.clone() for k, v in net.state_dict().items()},
                "physics": {k: v.clone() for k, v in physics.state_dict().items()},
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % cfg["print_every"] == 0 or epoch == 1:
            phys_vals = physics.get_all()
            print(f"  Epoch {epoch:4d} | Loss={loss.item():.4f} "
                  f"(D={loss_data.item():.4f} P={loss_physics.item():.4f} B={loss_boundary.item():.4f}) "
                  f"| Val MAE={val_mae:.4f} | K_La={phys_vals.get('K_La',0):.3f} "
                  f"R_fish={phys_vals.get('R_fish_base',0):.3f}")
        
        if patience_counter >= cfg["patience"]:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (patience={cfg['patience']})")
            break
    
    # 恢复最佳模型
    if best_state is not None:
        net.load_state_dict(best_state["net"])
        physics.load_state_dict(best_state["physics"])
    
    print(f"\n  ✅ 训练完成! Best Val MAE = {best_val_mae:.4f} mg/L")
    print(f"  学到的物理参数:")
    learned_params = physics.get_all()
    for k, v in learned_params.items():
        print(f"    {k} = {v:.4f}")
    
    return net, physics, df, input_cols, X_mean, X_std, history, learned_params


# ============================================================
# 可视化
# ============================================================
def plot_training_curves(history, site):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history["epoch"]
    
    # Loss curves
    axes[0].plot(epochs, history["loss"], "b-", alpha=0.5, label="Total")
    axes[0].plot(epochs, history["loss_data"], "r-", alpha=0.7, label="Data")
    axes[0].plot(epochs, history["loss_physics"], "g-", alpha=0.7, label="Physics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("训练损失")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    
    # Val MAE
    axes[1].plot(epochs, history["val_mae"], "purple")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (mg/L)")
    axes[1].set_title(f"验证集 MAE (Best={min(history['val_mae']):.4f})")
    axes[1].grid(True, alpha=0.3)
    
    # Physics loss ratio
    ratio = [p / (d + 1e-8) for p, d in zip(history["loss_physics"], history["loss_data"])]
    axes[2].plot(epochs, ratio, "orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Physics/Data 比值")
    axes[2].set_title("物理损失 / 数据损失")
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(f"PINN 训练过程 — {site}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / f"training_curves_{site}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_predictions(net, physics, df, input_cols, X_mean, X_std, site, device):
    """绘制预测 vs 真实对比图"""
    feat_cols = ["t_norm"] + input_cols
    X = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=device)
    y = df[TARGET_COL].values
    
    X_norm = (X - X_mean) / X_std
    
    net.eval()
    with torch.no_grad():
        y_pred = net(X_norm).cpu().numpy()
    
    # DO 饱和度
    T_water = df["水温_日均"].values
    DO_sat = 14.62 - 0.3898 * T_water + 0.006969 * T_water**2 - 5.897e-5 * T_water**3
    
    dates = pd.to_datetime(df["日期"])
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 上图：预测 vs 真实
    axes[0].plot(dates, y, "b-", alpha=0.6, linewidth=1, label="实测 DO")
    axes[0].plot(dates, y_pred, "r-", alpha=0.8, linewidth=1.5, label="PINN 预测")
    axes[0].plot(dates, DO_sat, "g--", alpha=0.4, linewidth=1, label="饱和 DO")
    axes[0].fill_between(dates, 0, 2, color="red", alpha=0.1, label="危险区 (<2 mg/L)")
    axes[0].set_ylabel("溶氧 (mg/L)")
    axes[0].set_title(f"PINN 溶氧预测 — {site}")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # 下图：残差
    residual = y - y_pred
    axes[1].bar(dates, residual, color="steelblue", alpha=0.6, width=1)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    axes[1].set_ylabel("残差 (mg/L)")
    axes[1].set_title(f"预测残差 | MAE={mae:.4f} RMSE={rmse:.4f}")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / f"prediction_{site}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return path, mae, rmse


def plot_counterfactual(net, physics, df, input_cols, X_mean, X_std, site, device):
    """反事实推理：温度升高 / 停止曝气 / 鱼密度翻倍"""
    feat_cols = ["t_norm"] + input_cols
    X_base = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=device)
    X_base_norm = (X_base - X_mean) / X_std
    
    idx_temp = feat_cols.index("水温_日均")
    dates = pd.to_datetime(df["日期"])
    
    net.eval()
    
    scenarios = {}
    
    # 基线
    with torch.no_grad():
        y_base = net(X_base_norm).cpu().numpy()
    scenarios["基线 (当前)"] = y_base
    
    # 场景1: 水温升高 3℃
    X_warm = X_base.clone()
    X_warm[:, idx_temp] += 3.0
    X_warm_norm = (X_warm - X_mean) / X_std
    with torch.no_grad():
        y_warm = net(X_warm_norm).cpu().numpy()
    scenarios["水温 +3℃"] = y_warm
    
    # 场景2: 水温降低 3℃
    X_cool = X_base.clone()
    X_cool[:, idx_temp] -= 3.0
    X_cool_norm = (X_cool - X_mean) / X_std
    with torch.no_grad():
        y_cool = net(X_cool_norm).cpu().numpy()
    scenarios["水温 -3℃"] = y_cool
    
    # 绘图
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = {"基线 (当前)": "blue", "水温 +3℃": "red", "水温 -3℃": "green"}
    for name, y_vals in scenarios.items():
        ax.plot(dates, y_vals, color=colors[name], alpha=0.7, linewidth=1.5, label=name)
    
    ax.fill_between(dates, 0, 2, color="red", alpha=0.1, label="危险区 (<2 mg/L)")
    ax.set_xlabel("日期")
    ax.set_ylabel("溶氧 (mg/L)")
    ax.set_title(f"反事实推理 — {site} | 水温变化对溶氧的影响")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = OUTPUT_DIR / f"counterfactual_{site}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return path, scenarios


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  🧬 PINN — 溶氧动力学模型")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  输出: {OUTPUT_DIR}")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    all_results = {}
    
    for site in ["红光", "喀左"]:
        print(f"\n\n{'='*60}")
        print(f"  📍 站点: {site}")
        print(f"{'='*60}")
        
        # 训练
        net, physics, df, input_cols, X_mean, X_std, history, learned_params = train_pinn(site)
        
        # 训练曲线
        curve_path = plot_training_curves(history, site)
        print(f"\n  📊 训练曲线: {curve_path}")
        
        # 预测图
        pred_path, mae, rmse = plot_predictions(
            net, physics, df, input_cols, X_mean, X_std, site, device
        )
        print(f"  📊 预测图: {pred_path}")
        print(f"  MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # 反事实推理
        cf_path, cf_scenarios = plot_counterfactual(
            net, physics, df, input_cols, X_mean, X_std, site, device
        )
        print(f"  📊 反事实推理: {cf_path}")
        
        # 保存模型
        model_path = OUTPUT_DIR / f"pinn_{site}.pt"
        torch.save({
            "net_state": net.state_dict(),
            "physics_state": physics.state_dict(),
            "X_mean": X_mean.cpu(),
            "X_std": X_std.cpu(),
            "input_cols": input_cols,
        }, model_path)
        print(f"  💾 模型: {model_path}")
        
        # 汇总
        all_results[site] = {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "best_val_mae": float(min(history["val_mae"])),
            "learned_params": learned_params,
            "n_data": len(df),
        }
    
    # 保存报告
    report = {
        "模型": "PINN (溶氧动力学)",
        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "物理方程": "dDO/dt = K_La*(DO_sat(T)-DO) - R_fish(T) - R_bio + P_photo",
        "结果": all_results,
    }
    report_path = OUTPUT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"  ✅ PINN 训练全部完成!")
    print(f"{'='*60}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {p.name:40s} ({p.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
