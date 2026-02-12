"""
配置文件：定义数据路径、特征列、标签列等参数
"""
from pathlib import Path

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件
FEATURED_HONGGUANG = DATA_DIR / "featured_红光.csv"
FEATURED_KAZUO = DATA_DIR / "featured_喀左.csv"
AUGMENTED_HONGGUANG = DATA_DIR / "augmented_红光.csv"
AUGMENTED_KAZUO = DATA_DIR / "augmented_喀左.csv"

# ============================================================
# 列定义
# ============================================================

# 非特征列（需排除）
META_COLS = [
    "日期", "基地", "温室", "模块", "温室_推断",
    "_is_augmented", "_scenario",
]

# 标签列（预测目标）
LABEL_COLS = [
    "蔬菜_事件数", "蔬菜_病害次数",
    "鱼_事件数", "鱼_死亡数量", "鱼_死亡重量_kg", "鱼_病害次数",
]

# 累积标签列（也需排除，因为它们是 label leakage）
CUMULATIVE_LABEL_COLS = [
    "蔬菜_病害次数_累积7d", "鱼_死亡数量_累积7d",
]

# 所有需排除的列
EXCLUDE_COLS = META_COLS + LABEL_COLS + CUMULATIVE_LABEL_COLS

# ============================================================
# 预测任务定义
# ============================================================

TASKS = {
    "蔬菜病害": {
        "label_col": "蔬菜_病害次数",
        "description": "蔬菜是否发生病害（二分类：有/无）",
    },
    "鱼类死亡": {
        "label_col": "鱼_死亡数量",
        "description": "鱼类是否发生死亡（二分类：有/无）",
    },
}

# ============================================================
# 模型超参数
# ============================================================

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
}

# 交叉验证折数
CV_FOLDS = 5

# SHAP 可视化展示的 top-N 特征数
SHAP_TOP_N = 20

# 随机种子
RANDOM_SEED = 42
