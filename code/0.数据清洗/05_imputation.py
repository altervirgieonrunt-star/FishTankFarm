"""
步骤5：缺失值分层插补
- 低缺失率列：前向/后向填充
- 高缺失率水质列：不做插补，添加 _有效 flag
"""

import pandas as pd
import numpy as np


# 低缺失率列：适合前向+后向填充
LOW_MISSING_COLS = [
    '最低水温℃', '最高水温℃', '水温_日均', '水温_std', '水温_日较差',
    '种植床1液位上限距种植床表面距离cm', '种植床2液位上限距种植床表面距离cm',
]

# 温室级列：可用同温室其他模块当日均值兜底
GREENHOUSE_LEVEL_COLS = [
    '最低气温℃', '最高气温℃', '气温_日均', '气温_std', '气温_日较差',
    '最低湿度%', '最高湿度%', '湿度_日均', '湿度_std',
    '能耗km/h', '光照时长h', 'DLI_approx', '光照_峰值',
]

# 高缺失率列：不做插补，仅添加 _有效 flag
HIGH_MISSING_COLS = [
    '溶氧mg/L', '氨氮mg/L', '亚盐mg/L', 'PH', 'EC值ms/cm',
]


def impute_single_base(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """对单个基地的合并数据执行缺失值插补"""
    df = df.copy()

    # 按模块排序（确保前向填充的时序正确性）
    df = df.sort_values(['模块', '日期']).reset_index(drop=True)

    # ── 1. 低缺失率列：按模块前向+后向填充 ──
    for col in LOW_MISSING_COLS:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        df[col] = df.groupby('模块')[col].transform(lambda s: s.ffill().bfill())
        # 极端情况：某模块全缺失 → 全局均值兜底
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
        after = df[col].isna().sum()
        if before > 0:
            print(f'    {col}: {before} → {after} 缺失')

    # ── 2. 种植床液位：字符串转数值后填充 ──
    for col in ['种植床1液位上限距种植床表面距离cm', '种植床2液位上限距种植床表面距离cm']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            before = df[col].isna().sum()
            df[col] = df.groupby('模块')[col].transform(lambda s: s.ffill().bfill())
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
            after = df[col].isna().sum()
            if before > 0:
                print(f'    {col}: {before} → {after} 缺失')

    # ── 3. 温室级列：同温室当日均值 → 前向填充 → 全局均值 ──
    # 推断温室列
    gh_col = '温室_推断' if '温室_推断' in df.columns else '温室'
    for col in GREENHOUSE_LEVEL_COLS:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        if before == 0:
            continue
        # 先用同温室同日均值填充
        gh_daily_mean = df.groupby(['日期', gh_col])[col].transform('mean')
        df[col] = df[col].fillna(gh_daily_mean)
        # 再按模块前向填充
        df[col] = df.groupby('模块')[col].transform(lambda s: s.ffill().bfill())
        # 最后全局均值兜底
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
        after = df[col].isna().sum()
        print(f'    {col}: {before} → {after} 缺失')

    # ── 4. 高缺失率水质列：添加 _有效 flag，不做插补 ──
    for col in HIGH_MISSING_COLS:
        if col not in df.columns:
            continue
        flag_col = col.split('mg/L')[0].split('ms/cm')[0].rstrip() + '_有效'
        # 简化 flag 列名
        flag_col = col.replace('mg/L', '').replace('ms/cm', '').replace('值', '').rstrip() + '_有效'
        df[flag_col] = df[col].notna().astype(int)
        n_valid = df[flag_col].sum()
        n_total = len(df)
        print(f'    {col}: 保留原值, 添加 {flag_col} ({n_valid}/{n_total} 有效, {n_valid/n_total*100:.1f}%)')

    return df


def impute_all(data: dict) -> dict:
    """对两个基地分别执行插补"""
    print('\n📦 步骤 5：缺失值分层插补')
    for base in ['红光', '喀左']:
        key = f'{base}_合并'
        if key not in data:
            print(f'  ⚠️ 跳过 {base}: 未找到合并数据')
            continue
        print(f'\n  📍 {base}基地 (共 {len(data[key])} 行):')
        data[key] = impute_single_base(data[key], base)

    return data


if __name__ == '__main__':
    from _01_load_and_fix import load_all
    from _02_parse_hourly import parse_all_hourly
    from _03_parse_disease import parse_all_disease
    from _04_time_align_merge import time_align_merge
    data = load_all()
    data = parse_all_hourly(data)
    data = parse_all_disease(data)
    data = time_align_merge(data)
    data = impute_all(data)
