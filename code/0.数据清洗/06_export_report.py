"""
步骤6：导出清洗后数据 + 质量报告
"""

import pandas as pd
import numpy as np
import os


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# 数值范围合理性检查规则
RANGE_CHECKS = {
    '水温_日均': (0, 45),
    '最低水温℃': (0, 45),
    '最高水温℃': (0, 45),
    '气温_日均': (-30, 55),
    '最低气温℃': (-30, 55),
    '最高气温℃': (-30, 55),
    '湿度_日均': (0, 100),
    '最低湿度%': (0, 100),
    '最高湿度%': (0, 100),
    '溶氧mg/L': (0, 25),
    '氨氮mg/L': (0, 50),
    'PH': (3, 11),
    'EC值ms/cm': (0, 20),
    '光照时长h': (0, 24),
}


def validate_and_export(data: dict):
    """验证数据质量并导出"""
    print('\n📦 步骤 6：验证与导出')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for base in ['红光', '喀左']:
        key = f'{base}_合并'
        if key not in data:
            continue

        df = data[key]
        print(f'\n  📍 {base}基地 — 最终数据: {df.shape[0]} 行, {df.shape[1]} 列')

        # ── 1. 主键完整性 ──
        print('\n  🔍 主键完整性:')
        for col in ['日期', '基地', '模块']:
            if col in df.columns:
                n_null = df[col].isna().sum()
                status = '✅' if n_null == 0 else '❌'
                print(f'    {status} {col}: {n_null} 个空值')

        # ── 2. 主键唯一性 ──
        dup_count = df.duplicated(subset=['日期', '模块']).sum()
        status = '✅' if dup_count == 0 else '❌'
        print(f'    {status} (日期, 模块) 重复行: {dup_count}')

        # ── 3. 缺失率统计 ──
        print('\n  📊 各列缺失率:')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            miss = df[col].isna().sum()
            rate = miss / len(df) * 100
            if miss > 0:
                bar = '█' * int(rate / 5) + '░' * (20 - int(rate / 5))
                print(f'    {col:30s}: {miss:>6d} ({rate:5.1f}%) {bar}')
        zero_miss = [c for c in numeric_cols if df[c].isna().sum() == 0]
        print(f'    ✅ {len(zero_miss)} 列零缺失: {zero_miss[:5]}{"..." if len(zero_miss) > 5 else ""}')

        # ── 4. 数值范围检查 ──
        print('\n  🔍 数值范围检查:')
        for col, (lo, hi) in RANGE_CHECKS.items():
            if col not in df.columns:
                continue
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            out_of_range = ((valid < lo) | (valid > hi)).sum()
            status = '✅' if out_of_range == 0 else '⚠️'
            print(f'    {status} {col}: [{valid.min():.2f}, {valid.max():.2f}]'
                  f' (期望 [{lo}, {hi}], 越界 {out_of_range} 条)')

        # ── 5. 时间连续性检查 ──
        print('\n  🔍 时间连续性 (抽查前 3 个模块):')
        modules = df['模块'].unique()[:3]
        for mod in modules:
            mod_df = df[df['模块'] == mod].sort_values('日期')
            dates = mod_df['日期'].dt.date
            if len(dates) < 2:
                continue
            gaps = pd.Series(dates.values[1:]) - pd.Series(dates.values[:-1])
            max_gap = gaps.max()
            n_gaps = (gaps > pd.Timedelta(days=1)).sum()
            print(f'    {mod}: {len(dates)} 天, 最大间隔 {max_gap.days}天, 间断 {n_gaps} 处')

        # ── 6. 导出 CSV ──
        out_path = os.path.join(OUTPUT_DIR, f'merged_{base}.csv')
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f'\n  💾 已导出: {out_path} ({size_mb:.1f} MB)')

        # ── 7. 列清单 ──
        print(f'\n  📋 最终列清单 ({len(df.columns)} 列):')
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            print(f'    {i+1:>3d}. {col} ({dtype})')


if __name__ == '__main__':
    from _01_load_and_fix import load_all
    from _02_parse_hourly import parse_all_hourly
    from _03_parse_disease import parse_all_disease
    from _04_time_align_merge import time_align_merge
    from _05_imputation import impute_all
    data = load_all()
    data = parse_all_hourly(data)
    data = parse_all_disease(data)
    data = time_align_merge(data)
    data = impute_all(data)
    validate_and_export(data)
