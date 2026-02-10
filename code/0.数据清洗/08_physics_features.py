"""
步骤8：物理特征工程
- 变化率 (diff) 特征
- 溶氧饱和度 (基于水温计算理论 DO 饱和值)
- 温差耦合项 (水温-气温差)
- 滞后特征 (lag)
- 滚动窗口统计 (rolling)

解决审查反馈: "物理特征提取不足" + "时序窗口构造缺失"
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def calc_do_saturation(water_temp: pd.Series) -> pd.Series:
    """
    基于水温计算理论溶氧饱和度 (mg/L)
    公式来源: Benson & Krause (1984) 简化版
    DO_sat = 14.62 - 0.3898·T + 0.006969·T² - 0.00005896·T³
    适用于标准大气压、纯水条件
    """
    T = water_temp.clip(lower=0, upper=45)
    do_sat = 14.62 - 0.3898 * T + 0.006969 * T**2 - 0.00005896 * T**3
    return do_sat.clip(lower=0)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """为单个基地的数据添加物理特征"""
    df = df.copy()
    df = df.sort_values(['模块', '日期']).reset_index(drop=True)

    # ── 1. 溶氧饱和度相关 ──
    if '水温_日均' in df.columns:
        df['DO_饱和度_理论'] = calc_do_saturation(df['水温_日均'])

    if '溶氧mg/L' in df.columns and 'DO_饱和度_理论' in df.columns:
        # 溶氧饱和比: 实测DO / 理论饱和DO (>1 表示过饱和)
        df['DO_饱和比'] = df['溶氧mg/L'] / df['DO_饱和度_理论'].replace(0, np.nan)
        # DO亏损: 理论值 - 实测值 (>0 表示缺氧)
        df['DO_亏损'] = df['DO_饱和度_理论'] - df['溶氧mg/L']

    # ── 2. 温差耦合项 ──
    if '水温_日均' in df.columns and '气温_日均' in df.columns:
        df['水气温差'] = df['水温_日均'] - df['气温_日均']

    # ── 3. 变化率 (diff) 特征 — 按模块计算 ──
    diff_cols = {
        '水温_日均': '水温_变化率',
        '气温_日均': '气温_变化率',
        '湿度_日均': '湿度_变化率',
        '溶氧mg/L': '溶氧_变化率',
        '氨氮mg/L': '氨氮_变化率',
        'PH': 'PH_变化率',
    }
    for src, dst in diff_cols.items():
        if src in df.columns:
            df[dst] = df.groupby('模块')[src].diff()

    # ── 4. 滞后特征 (lag) — 按模块计算 ──
    lag_cols = ['水温_日均', '气温_日均', '湿度_日均']
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in [1, 3]:
            df[f'{col}_lag{lag}d'] = df.groupby('模块')[col].shift(lag)

    # ── 5. 滚动窗口统计 — 按模块计算 ──
    rolling_cols = ['水温_日均', '气温_日均', '湿度_日均']
    for col in rolling_cols:
        if col not in df.columns:
            continue
        for window in [3, 7]:
            grouped = df.groupby('模块')[col]
            df[f'{col}_roll{window}d_mean'] = grouped.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df[f'{col}_roll{window}d_std'] = grouped.transform(
                lambda s: s.rolling(window, min_periods=1).std()
            )

    # ── 6. 累积病害压力 (过去7天病害事件总数) ──
    for col in ['蔬菜_病害次数', '鱼_死亡数量']:
        if col in df.columns:
            df[f'{col}_累积7d'] = df.groupby('模块')[col].transform(
                lambda s: s.rolling(7, min_periods=1).sum()
            )

    return df


def run_physics_features():
    """对两个基地的清洗后数据添加物理特征"""
    print('=' * 70)
    print('🔬 物理特征工程')
    print('=' * 70)

    for base in ['红光', '喀左']:
        in_path = os.path.join(DATA_DIR, f'cleaned_{base}.csv')
        if not os.path.exists(in_path):
            print(f'  ❌ 未找到 {in_path}')
            continue

        df = pd.read_csv(in_path, parse_dates=['日期'])
        n_cols_before = len(df.columns)

        print(f'\n  📍 {base}基地 ({len(df)} 行, {n_cols_before} 列):')
        df = add_physics_features(df)
        n_cols_after = len(df.columns)

        new_cols = [c for c in df.columns if c not in pd.read_csv(in_path, nrows=0).columns]
        print(f'    → 新增 {n_cols_after - n_cols_before} 个特征:')
        for c in new_cols:
            n_valid = df[c].notna().sum()
            print(f'      {c}: {n_valid}/{len(df)} 有效 ({n_valid/len(df)*100:.1f}%)')

        # 导出
        out_path = os.path.join(DATA_DIR, f'featured_{base}.csv')
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f'\n  💾 已导出: {out_path} ({size_mb:.1f} MB, {len(df.columns)} 列)')

    print(f'\n{"=" * 70}')
    print('✅ 物理特征工程完成!')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    run_physics_features()
