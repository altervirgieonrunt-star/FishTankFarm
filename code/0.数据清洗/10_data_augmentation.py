"""
步骤10：数据合成与增强
- 针对稀少的病害正样本进行物理扰动增强
- 基于领域知识的合成场景生成

解决审查反馈: "数据合成与增强"（文档4.2节）
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# 扰动范围 (基于物理合理性)
PERTURBATION_CONFIG = {
    '水温_日均':    {'std': 1.5,  'min': 0,  'max': 40},
    '气温_日均':    {'std': 2.0,  'min': -20, 'max': 50},
    '湿度_日均':    {'std': 5.0,  'min': 0,  'max': 100},
    '溶氧mg/L':    {'std': 0.5,  'min': 0,  'max': 20},
    '氨氮mg/L':    {'std': 0.3,  'min': 0,  'max': 10},
    'PH':          {'std': 0.2,  'min': 4,  'max': 10},
    '光照时长h':   {'std': 0.5,  'min': 0,  'max': 24},
    '能耗km/h':    {'std': 5.0,  'min': 0,  'max': 500},
}


def augment_positive_samples(df: pd.DataFrame, target_col: str,
                              n_augments: int = 5,
                              random_seed: int = 42) -> pd.DataFrame:
    """
    对正样本（target_col > 0）进行物理扰动增强

    策略:
    1. 筛选出正样本（有病害/死亡的行）
    2. 对每条正样本生成 n_augments 个扰动副本
    3. 扰动方式: 高斯噪声 + 物理边界裁剪
    """
    rng = np.random.RandomState(random_seed)
    positive = df[df[target_col] > 0].copy()
    n_pos = len(positive)

    if n_pos == 0:
        print(f'    ⚠️ {target_col}: 无正样本，跳过增强')
        return pd.DataFrame()

    augmented_rows = []
    for _ in range(n_augments):
        aug = positive.copy()
        aug['_is_augmented'] = True

        # 对数值列施加高斯扰动
        for col, cfg in PERTURBATION_CONFIG.items():
            if col not in aug.columns:
                continue
            valid_mask = aug[col].notna()
            noise = rng.normal(0, cfg['std'], size=valid_mask.sum())
            aug.loc[valid_mask, col] = (
                aug.loc[valid_mask, col] + noise
            ).clip(lower=cfg['min'], upper=cfg['max'])

        augmented_rows.append(aug)

    result = pd.concat(augmented_rows, ignore_index=True)
    print(f'    ✅ {target_col}: {n_pos} 条正样本 × {n_augments} = {len(result)} 条增强数据')
    return result


def generate_anomaly_scenarios(df: pd.DataFrame, n_scenarios: int = 100,
                                random_seed: int = 123) -> pd.DataFrame:
    """
    生成异常场景模拟数据
    模拟: 曝气故障(DO骤降)、过量投喂(氨氮飙升)、高温胁迫
    """
    rng = np.random.RandomState(random_seed)
    base_rows = df.sample(n=min(n_scenarios, len(df)), random_state=rng).copy()
    scenarios = []

    # 场景1: 曝气故障 → 溶氧骤降至 2-4 mg/L
    if '溶氧mg/L' in base_rows.columns:
        s1 = base_rows.copy()
        s1['溶氧mg/L'] = rng.uniform(1.5, 4.0, size=len(s1))
        s1['_scenario'] = '曝气故障'
        s1['_is_augmented'] = True
        scenarios.append(s1)

    # 场景2: 过量投喂 → 氨氮升高至 2-6 mg/L
    if '氨氮mg/L' in base_rows.columns:
        s2 = base_rows.copy()
        s2['氨氮mg/L'] = rng.uniform(2.0, 6.0, size=len(s2))
        s2['_scenario'] = '过量投喂'
        s2['_is_augmented'] = True
        scenarios.append(s2)

    # 场景3: 高温胁迫 → 水温升至 32-38°C
    if '水温_日均' in base_rows.columns:
        s3 = base_rows.copy()
        s3['水温_日均'] = rng.uniform(32, 38, size=len(s3))
        s3['_scenario'] = '高温胁迫'
        s3['_is_augmented'] = True
        scenarios.append(s3)

    if scenarios:
        result = pd.concat(scenarios, ignore_index=True)
        print(f'    ✅ 异常场景模拟: 生成 {len(result)} 条 '
              f'({len(scenarios)} 种场景 × ~{n_scenarios} 条)')
        return result
    return pd.DataFrame()


def run_augmentation():
    """对两个基地执行数据增强"""
    print('=' * 70)
    print('🧬 数据合成与增强')
    print('=' * 70)

    for base in ['红光', '喀左']:
        in_path = os.path.join(DATA_DIR, f'cleaned_{base}.csv')
        if not os.path.exists(in_path):
            print(f'  ❌ 未找到 {in_path}')
            continue

        df = pd.read_csv(in_path, parse_dates=['日期'])
        df['_is_augmented'] = False
        df['_scenario'] = '真实数据'

        print(f'\n  📍 {base}基地 (原始 {len(df)} 行):')

        # 1. 正样本增强
        aug_veg = augment_positive_samples(df, '蔬菜_病害次数', n_augments=5)
        aug_fish = augment_positive_samples(df, '鱼_死亡数量', n_augments=5)

        # 2. 异常场景模拟
        aug_scenarios = generate_anomaly_scenarios(df, n_scenarios=200)

        # 3. 合并
        parts = [df]
        if len(aug_veg) > 0:
            parts.append(aug_veg)
        if len(aug_fish) > 0:
            parts.append(aug_fish)
        if len(aug_scenarios) > 0:
            parts.append(aug_scenarios)

        augmented = pd.concat(parts, ignore_index=True)

        # 导出
        out_path = os.path.join(DATA_DIR, f'augmented_{base}.csv')
        augmented.to_csv(out_path, index=False, encoding='utf-8-sig')
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        n_real = len(df)
        n_aug = len(augmented) - n_real
        print(f'\n    📊 合计: {n_real} 真实 + {n_aug} 增强 = {len(augmented)} 行')
        print(f'    💾 已导出: {out_path} ({size_mb:.1f} MB)')

    print(f'\n{"=" * 70}')
    print('✅ 数据合成与增强完成!')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    run_augmentation()
