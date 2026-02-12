"""
PINN 配置文件
- 数据路径
- 物理参数初始值与约束
- 网络与训练超参数
"""
from pathlib import Path

# ============================================================
# 路径
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据配置
# ============================================================
# 输入特征列（来自 cleaned_*.csv）
INPUT_COLS = [
    "水温_日均",       # ℃
    "气温_日均",       # ℃
    "光照时长h",       # h
]

# 目标列
TARGET_COL = "溶氧mg/L"    # mg/L

# 可选辅助列（如存在则加入）
AUX_COLS = [
    "能耗km/h",        # 能耗（间接反映曝气强度）
    "氨氮_有效",       # 0/1 标记
]

RANDOM_SEED = 42

# ============================================================
# 物理参数初始值
# ============================================================
PHYSICS_PARAMS = {
    # 复氧传质系数 (day^-1)
    # 典型值: 1~10, 取决于曝气方式
    "K_La": 2.0,

    # 鱼呼吸耗氧基础速率 (mg/L/day)
    # 典型值: 0.5~3.0
    "R_fish_base": 1.0,

    # 耗氧温度系数 (无量纲)
    # R_fish = R_fish_base * (1 + alpha_T * (T - T_ref))
    "alpha_T": 0.05,

    # 参考温度 (℃)
    "T_ref": 25.0,

    # 微生物耗氧速率 (mg/L/day)
    "R_bio": 0.5,

    # 光合产氧速率 (mg/L/day per hour of light)
    "P_photo_rate": 0.1,
}

# 参数约束（下界, 上界）
PARAM_BOUNDS = {
    "K_La": (0.01, 20.0),
    "R_fish_base": (0.01, 10.0),
    "alpha_T": (0.001, 0.3),
    "R_bio": (0.01, 5.0),
    "P_photo_rate": (0.001, 1.0),
}

# ============================================================
# 网络超参数
# ============================================================
NET_CONFIG = {
    "hidden_layers": [64, 64, 64],
    "activation": "tanh",
    "dropout": 0.0,
}

# ============================================================
# 训练超参数
# ============================================================
TRAIN_CONFIG = {
    "epochs": 3000,
    "lr_net": 1e-3,          # 网络权重学习率
    "lr_physics": 1e-2,      # 物理参数学习率（更大以便快速收敛）
    "batch_size": 64,
    "lambda_physics": 1.0,   # 物理损失权重（初始值）
    "lambda_boundary": 0.1,  # 边界损失权重
    "val_ratio": 0.2,        # 验证集比例
    "patience": 300,         # Early stopping patience
    "scheduler_step": 500,
    "scheduler_gamma": 0.5,
    "print_every": 100,
}
