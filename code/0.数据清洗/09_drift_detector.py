"""
步骤9：传感器漂移监测
- 基于 Z-score 的实时漂移检测
- 生成漂移报告 CSV

解决审查反馈: "增加传感器漂移监测" + "商业化叙事素材"
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# 监测的传感器列
MONITOR_COLS = [
    '水温_日均', '气温_日均', '湿度_日均',
    '溶氧mg/L', '氨氮mg/L', 'PH', 'EC值ms/cm',
    '能耗km/h',
]

ZSCORE_THRESHOLD = 3.0  # Z-score 阈值


def detect_drift(df: pd.DataFrame, base: str) -> pd.DataFrame:
    """
    基于滑动窗口 Z-score 检测传感器漂移
    对每个 (模块) 分组计算 30 天滑动窗口的 Z-score
    """
    df = df.copy()
    df = df.sort_values(['模块', '日期']).reset_index(drop=True)

    drift_records = []

    for col in MONITOR_COLS:
        if col not in df.columns:
            continue

        valid_mask = df[col].notna()
        if valid_mask.sum() < 30:
            continue

        # 全局 Z-score
        global_mean = df.loc[valid_mask, col].mean()
        global_std = df.loc[valid_mask, col].std()
        if global_std == 0:
            continue

        z_global = ((df[col] - global_mean) / global_std).abs()

        # 滑动窗口 Z-score (30天窗口, 按模块)
        rolling_mean = df.groupby('模块')[col].transform(
            lambda s: s.rolling(30, min_periods=7).mean()
        )
        rolling_std = df.groupby('模块')[col].transform(
            lambda s: s.rolling(30, min_periods=7).std()
        )
        z_rolling = ((df[col] - rolling_mean) / rolling_std.replace(0, np.nan)).abs()

        # 标记漂移点: 全局 Z > 阈值 或 滑动 Z > 阈值
        drift_mask = valid_mask & ((z_global > ZSCORE_THRESHOLD) | (z_rolling > ZSCORE_THRESHOLD))
        n_drift = drift_mask.sum()

        if n_drift > 0:
            drift_rows = df.loc[drift_mask, ['日期', '基地', '模块', col]].copy()
            drift_rows['指标'] = col
            drift_rows['Z_global'] = z_global[drift_mask].values
            drift_rows['Z_rolling'] = z_rolling[drift_mask].values
            drift_rows['实测值'] = df.loc[drift_mask, col].values
            drift_rows['全局均值'] = global_mean
            drift_rows['全局标准差'] = global_std
            drift_records.append(drift_rows[['日期', '基地', '模块', '指标',
                                              '实测值', 'Z_global', 'Z_rolling',
                                              '全局均值', '全局标准差']])

    if drift_records:
        report = pd.concat(drift_records, ignore_index=True)
    else:
        report = pd.DataFrame(columns=['日期', '基地', '模块', '指标',
                                        '实测值', 'Z_global', 'Z_rolling',
                                        '全局均值', '全局标准差'])
    return report


def run_drift_detection():
    """对两个基地生成漂移报告"""
    print('=' * 70)
    print('📡 传感器漂移监测')
    print('=' * 70)

    all_reports = []
    for base in ['红光', '喀左']:
        in_path = os.path.join(DATA_DIR, f'cleaned_{base}.csv')
        if not os.path.exists(in_path):
            print(f'  ❌ 未找到 {in_path}')
            continue

        df = pd.read_csv(in_path, parse_dates=['日期'])
        print(f'\n  📍 {base}基地 ({len(df)} 行):')

        report = detect_drift(df, base)
        all_reports.append(report)

        # 按指标统计
        if len(report) > 0:
            summary = report.groupby('指标').agg(
                漂移点数=('实测值', 'count'),
                涉及模块数=('模块', 'nunique'),
                最早日期=('日期', 'min'),
                最晚日期=('日期', 'max'),
            ).reset_index()
            print(f'    ⚠️ 检测到 {len(report)} 个漂移点:')
            for _, row in summary.iterrows():
                print(f'      {row["指标"]}: {row["漂移点数"]}个点, '
                      f'{row["涉及模块数"]}个模块, '
                      f'{row["最早日期"].strftime("%Y-%m-%d")} ~ {row["最晚日期"].strftime("%Y-%m-%d")}')
        else:
            print('    ✅ 未检测到显著漂移')

    # 合并导出
    if all_reports:
        full_report = pd.concat(all_reports, ignore_index=True)
        out_path = os.path.join(DATA_DIR, 'drift_report.csv')
        full_report.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f'\n  💾 漂移报告已导出: {out_path} ({len(full_report)} 条记录)')

    print(f'\n{"=" * 70}')
    print('✅ 传感器漂移监测完成!')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    run_drift_detection()
