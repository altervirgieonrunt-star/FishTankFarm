"""
步骤4：时间对齐与多表 JOIN
- 以模块环境日数据为主时间轴 (日期 × 模块)
- LEFT JOIN 温室环境日数据 (日期 × 温室)
- LEFT JOIN 发病事件日聚合 (日期 × 模块)
"""

import pandas as pd
import re


def extract_greenhouse_from_module(module_name: str, base: str) -> str:
    """
    从模块名推断温室名
    红光: "红光1-2" → "红光1号温室", "红光10-1" → "红光10号温室"
    喀左: "喀左11-1" → "喀左11号棚", "喀左0-1小" → "喀左0号棚"
    """
    if base == '红光' or base == '天津红光基地':
        m = re.match(r'红光(\d+)-', module_name)
        if m:
            return f'红光{m.group(1)}号温室'
    elif base == '喀左' or base == '辽宁喀左基地':
        m = re.match(r'喀左(\d+)-', module_name)
        if m:
            return f'喀左{m.group(1)}号棚'
    return None


def merge_single_base(
    module_env: pd.DataFrame,
    greenhouse_env: pd.DataFrame,
    veg_daily: pd.DataFrame,
    fish_daily: pd.DataFrame,
    base_short: str,  # '红光' | '喀左'
) -> pd.DataFrame:
    """合并单个基地的全部数据"""

    # 1. 为模块环境数据推断温室名
    module_env = module_env.copy()
    module_env['温室_推断'] = module_env['模块'].apply(
        lambda x: extract_greenhouse_from_module(x, base_short)
    )

    # 验证推断结果
    n_none = module_env['温室_推断'].isna().sum()
    if n_none > 0:
        failed = module_env[module_env['温室_推断'].isna()]['模块'].unique()
        print(f'    ⚠️ 未能推断温室名的模块 ({n_none} 行): {list(failed)[:5]}')

    # 2. LEFT JOIN 温室环境
    greenhouse_env = greenhouse_env.copy()
    # 温室环境中的温室列名就是原始的 "温室" 列
    gh_merge_key = greenhouse_env['温室'].values[0]  # 检测温室列的命名风格
    # 重命名温室环境的温室列以匹配
    greenhouse_env = greenhouse_env.rename(columns={'温室': '温室_gh'})

    merged = module_env.merge(
        greenhouse_env,
        left_on=['日期', '温室_推断'],
        right_on=['日期', '温室_gh'],
        how='left',
        suffixes=('', '_温室')
    )

    # 清理冗余列
    drop_cols = [c for c in ['基地_温室', '温室_gh'] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    print(f'    ✅ 合并温室环境: {len(module_env)} → {len(merged)} 行')

    # 3. LEFT JOIN 蔬菜发病日聚合
    if len(veg_daily) > 0:
        merged = merged.merge(
            veg_daily,
            on=['日期', '模块'],
            how='left',
            suffixes=('', '_蔬菜日聚合')
        )
        # 填充无事件日为 0
        for col in ['蔬菜_事件数', '蔬菜_病害次数']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)

        # 清理冗余基地列
        drop_cols = [c for c in merged.columns if c.endswith('_蔬菜日聚合')]
        merged = merged.drop(columns=drop_cols)
    else:
        merged['蔬菜_事件数'] = 0
        merged['蔬菜_病害次数'] = 0

    # 4. LEFT JOIN 鱼类发病日聚合
    if len(fish_daily) > 0:
        merged = merged.merge(
            fish_daily,
            on=['日期', '模块'],
            how='left',
            suffixes=('', '_鱼类日聚合')
        )
        for col in ['鱼_事件数', '鱼_死亡数量', '鱼_病害次数']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)
        for col in ['鱼_死亡重量_kg']:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        drop_cols = [c for c in merged.columns if c.endswith('_鱼类日聚合')]
        merged = merged.drop(columns=drop_cols)
    else:
        merged['鱼_事件数'] = 0
        merged['鱼_死亡数量'] = 0
        merged['鱼_死亡重量_kg'] = 0.0
        merged['鱼_病害次数'] = 0

    return merged


def time_align_merge(data: dict) -> dict:
    """对两个基地分别执行合并"""
    print('\n📦 步骤 4：时间对齐与多表合并')
    for base in ['红光', '喀左']:
        print(f'\n  📍 {base}基地:')
        merged = merge_single_base(
            module_env=data[f'{base}_模块环境'],
            greenhouse_env=data[f'{base}_温室环境'],
            veg_daily=data.get(f'{base}_蔬菜日聚合', pd.DataFrame()),
            fish_daily=data.get(f'{base}_鱼类日聚合', pd.DataFrame()),
            base_short=base,
        )
        data[f'{base}_合并'] = merged
        print(f'    → 最终合并表: {merged.shape[0]} 行, {merged.shape[1]} 列')

    return data


if __name__ == '__main__':
    from _01_load_and_fix import load_all
    from _02_parse_hourly import parse_all_hourly
    from _03_parse_disease import parse_all_disease
    data = load_all()
    data = parse_all_hourly(data)
    data = parse_all_disease(data)
    data = time_align_merge(data)
