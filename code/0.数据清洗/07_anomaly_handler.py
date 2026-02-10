#!/usr/bin/env python3
"""
异常值检测与处理 (Winsorization + 标记)
基于 IQR (四分位距) 和领域知识的双重异常值处理

运行方式:
    source code/.venv/bin/activate
    python code/0.数据清洗/07_anomaly_handler.py
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# ──────────────── 领域知识硬边界 ────────────────
# 超出此范围的值一定是传感器故障，直接标记为无效
HARD_BOUNDS = {
    '最低水温℃':   (0, 45),
    '最高水温℃':   (0, 45),
    '水温_日均':    (0, 45),
    '水温_std':     (0, 20),
    '水温_日较差':  (0, 30),
    '最低气温℃':   (-30, 55),
    '最高气温℃':   (-30, 55),
    '气温_日均':    (-30, 55),
    '气温_std':     (0, 25),
    '气温_日较差':  (0, 40),
    '最低湿度%':    (0, 100),
    '最高湿度%':    (0, 100),
    '湿度_日均':    (0, 100),
    '溶氧mg/L':    (0, 25),
    '氨氮mg/L':    (0, 50),
    '亚盐mg/L':    (0, 50),
    'PH':          (3, 11),
    'EC值ms/cm':   (0, 20),
    '能耗km/h':    (0, 500),
    '光照时长h':   (0, 24),
    'DLI_approx':  (0, 5000000),
    '光照_峰值':   (0, 200000),
}


def detect_and_handle_anomalies(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    异常值处理流程：
    1. 硬边界截断：超出物理可能范围的 → 置为 NaN
    2. IQR Winsorization：在合理范围内的极端值 → 截断到 [Q1-1.5*IQR, Q3+1.5*IQR]
    3. 重新填充因截断产生的 NaN（前向+后向填充）
    """
    df = df.copy()
    total_clipped = 0
    total_hard_nan = 0

    print(f'\n  📍 {base}基地 ({len(df)} 行):')

    # ── 步骤 1：硬边界 → 置为 NaN ──
    print('\n  ── 硬边界截断 (传感器故障) ──')
    for col, (lo, hi) in HARD_BOUNDS.items():
        if col not in df.columns:
            continue
        valid_before = df[col].notna()
        out_of_range = valid_before & ((df[col] < lo) | (df[col] > hi))
        n_out = out_of_range.sum()
        if n_out > 0:
            # 记录被标记的异常值范围
            anomaly_vals = df.loc[out_of_range, col]
            print(f'    ⚠️ {col}: {n_out} 条越界 '
                  f'(范围 [{anomaly_vals.min():.2f}, {anomaly_vals.max():.2f}], '
                  f'有效范围 [{lo}, {hi}]) → 置为 NaN')
            df.loc[out_of_range, col] = np.nan
            total_hard_nan += n_out

    if total_hard_nan == 0:
        print('    ✅ 无硬边界越界')

    # ── 步骤 2：IQR Winsorization ──
    print('\n  ── IQR Winsorization (统计极端值) ──')
    # 只对连续传感器数据做 IQR（不对稀疏的水质列做）
    iqr_cols = [
        '最低水温℃', '最高水温℃', '水温_日均', '水温_std', '水温_日较差',
        '最低气温℃', '最高气温℃', '气温_日均', '气温_std', '气温_日较差',
        '最低湿度%', '最高湿度%', '湿度_日均', '湿度_std',
        '能耗km/h', '光照时长h', 'DLI_approx', '光照_峰值',
        '种植床1液位上限距种植床表面距离cm', '种植床2液位上限距种植床表面距离cm',
    ]

    for col in iqr_cols:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) < 10:
            continue

        q1 = valid.quantile(0.01)
        q99 = valid.quantile(0.99)

        below = (df[col] < q1).sum()
        above = (df[col] > q99).sum()
        n_clip = below + above

        if n_clip > 0:
            df[col] = df[col].clip(lower=q1, upper=q99)
            total_clipped += n_clip
            print(f'    📏 {col}: 截断 {n_clip} 条到 [{q1:.2f}, {q99:.2f}]')

    if total_clipped == 0:
        print('    ✅ 无 IQR 截断')

    # ── 步骤 3：重新填充因硬边界置 NaN 产生的缺失 ──
    if total_hard_nan > 0:
        print('\n  ── 重新填充因截断产生的 NaN ──')
        df = df.sort_values(['模块', '日期']).reset_index(drop=True)
        refill_cols = [col for col in HARD_BOUNDS.keys() if col in df.columns
                       and col not in ['溶氧mg/L', '氨氮mg/L', '亚盐mg/L', 'PH', 'EC值ms/cm']]
        for col in refill_cols:
            before = df[col].isna().sum()
            if before == 0:
                continue
            df[col] = df.groupby('模块')[col].transform(lambda s: s.ffill().bfill())
            after = df[col].isna().sum()
            if before != after:
                print(f'    🔄 {col}: {before} → {after} 缺失 (前向+后向填充)')

    print(f'\n  📊 总计: 硬边界置NaN {total_hard_nan} 条, IQR截断 {total_clipped} 条')
    return df


def run_anomaly_handling():
    """对两个基地的清洗后数据执行异常值处理"""
    print('=' * 70)
    print('🔍 异常值检测与处理')
    print('=' * 70)

    for base in ['红光', '喀左']:
        in_path = os.path.join(DATA_DIR, f'merged_{base}.csv')
        if not os.path.exists(in_path):
            print(f'  ❌ 未找到 {in_path}')
            continue

        df = pd.read_csv(in_path, parse_dates=['日期'])
        df = detect_and_handle_anomalies(df, base)

        # 导出处理后数据
        out_path = os.path.join(DATA_DIR, f'cleaned_{base}.csv')
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f'\n  💾 已导出: {out_path} ({size_mb:.1f} MB)')

        # 最终范围检查
        print(f'\n  🔍 处理后数值范围:')
        for col, (lo, hi) in HARD_BOUNDS.items():
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            out = ((valid < lo) | (valid > hi)).sum()
            status = '✅' if out == 0 else '❌'
            print(f'    {status} {col}: [{valid.min():.2f}, {valid.max():.2f}]')

    print(f'\n{"=" * 70}')
    print('✅ 异常值处理完成!')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    run_anomaly_handling()
