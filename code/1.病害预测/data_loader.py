"""
数据加载与预处理模块
- 加载 featured CSV
- 拆分特征 / 标签
- 处理缺失值
- 二值化标签（有/无病害）
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    FEATURED_HONGGUANG, FEATURED_KAZUO,
    AUGMENTED_HONGGUANG, AUGMENTED_KAZUO,
    EXCLUDE_COLS, RANDOM_SEED,
)


def load_featured(path, site_name: str = "") -> pd.DataFrame:
    """加载一个 featured CSV 文件"""
    df = pd.read_csv(path, parse_dates=["日期"])
    print(f"[{site_name}] 加载 {path.name}: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def prepare_features_labels(df: pd.DataFrame, label_col: str):
    """
    从 DataFrame 中拆分特征和标签。
    标签二值化：> 0 → 1（有病害/死亡），= 0 → 0（无）

    Returns:
        X: 特征 DataFrame
        y: 二值化标签 Series
        feature_names: 特征名列表
    """
    # 排除非特征列
    exclude = [c for c in EXCLUDE_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    X = df[feature_cols].copy()
    y = (df[label_col] > 0).astype(int)

    # 处理缺失值：用该列中位数填充
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # 确保没有 inf
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(0)

    print(f"  特征数: {X.shape[1]}, 正样本: {y.sum()} ({y.mean()*100:.1f}%), "
          f"负样本: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    return X, y, list(X.columns)


def load_train_test(task_config: dict, use_augmented: bool = False):
    """
    加载红光（训练）和喀左（测试）数据

    Args:
        task_config: 任务配置 dict，包含 label_col
        use_augmented: 是否使用增强数据

    Returns:
        X_train, X_val, y_train, y_val: 红光数据的训练/验证集
        X_test, y_test: 喀左数据（独立测试集）
        feature_names: 特征名列表
    """
    label_col = task_config["label_col"]

    # 加载数据
    if use_augmented:
        df_hg = load_featured(AUGMENTED_HONGGUANG, "红光-增强")
        # 增强数据可能有 _is_augmented 列
        if "_is_augmented" in df_hg.columns:
            print(f"  增强数据中真实样本: {(~df_hg['_is_augmented']).sum()}, "
                  f"合成样本: {df_hg['_is_augmented'].sum()}")
    else:
        df_hg = load_featured(FEATURED_HONGGUANG, "红光")

    df_kz = load_featured(FEATURED_KAZUO, "喀左")

    # 准备特征和标签
    print("\n📊 红光数据（训练集）:")
    X_hg, y_hg, feature_names = prepare_features_labels(df_hg, label_col)

    print("\n📊 喀左数据（独立测试集）:")
    X_kz, y_kz, _ = prepare_features_labels(df_kz, label_col)

    # 确保喀左数据和红光数据使用相同特征
    common_features = [f for f in feature_names if f in X_kz.columns]
    missing_in_kz = [f for f in feature_names if f not in X_kz.columns]
    if missing_in_kz:
        print(f"\n  ⚠️ 喀左数据缺少 {len(missing_in_kz)} 个特征: {missing_in_kz}")
        for f in missing_in_kz:
            X_kz[f] = 0

    X_hg = X_hg[feature_names]
    X_kz = X_kz[feature_names]

    # 红光数据拆分为训练集和验证集 (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_hg, y_hg, test_size=0.2, random_state=RANDOM_SEED, stratify=y_hg
    )

    print(f"\n✅ 数据准备完成:")
    print(f"  训练集: {X_train.shape[0]} 行")
    print(f"  验证集: {X_val.shape[0]} 行")
    print(f"  独立测试集 (喀左): {X_kz.shape[0]} 行")

    return X_train, X_val, y_train, y_val, X_kz, y_kz, feature_names


if __name__ == "__main__":
    from config import TASKS
    for task_name, task_cfg in TASKS.items():
        print(f"\n{'='*60}")
        print(f"任务: {task_name} — {task_cfg['description']}")
        print(f"{'='*60}")
        load_train_test(task_cfg, use_augmented=False)
