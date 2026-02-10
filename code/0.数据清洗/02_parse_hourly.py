"""
步骤2：解析逐小时字符串列 → 数值统计量
格式示例: "00=16.0,01=15.9,...,23=14.4,"
"""

import pandas as pd
import numpy as np
import re


def parse_hourly_string(s: str) -> list[float]:
    """
    解析 "00=16.0,01=15.9,...,23=14.4," → [16.0, 15.9, ..., 14.4]
    返回长度为 24 的浮点数列表；解析失败返回 [NaN]*24
    """
    if not isinstance(s, str) or not s.strip():
        return [np.nan] * 24

    values = [np.nan] * 24
    # 匹配 HH=数值 的模式
    for match in re.finditer(r'(\d{2})=([\d.]+)', s):
        hour = int(match.group(1))
        val = float(match.group(2))
        if 0 <= hour <= 23:
            values[hour] = val
    return values


def compute_daily_stats(hourly_values: list[float]) -> dict:
    """从 24 小时数据计算日统计量"""
    arr = np.array(hourly_values)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {'日均': np.nan, 'std': np.nan, '日较差': np.nan}
    return {
        '日均': float(np.mean(valid)),
        'std': float(np.std(valid)),
        '日较差': float(np.max(valid) - np.min(valid)),
    }


def compute_light_stats(hourly_values: list[float]) -> dict:
    """光照的日统计量：日累积光照近似值 (DLI) + 峰值"""
    arr = np.array(hourly_values)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {'DLI_approx': np.nan, '峰值': np.nan}
    # DLI 近似: 逐小时累加 (单位: Lux·h)
    # 注意：真正的 DLI 应基于 PAR (μmol/m²/s)，这里用 Lux 近似
    dli = float(np.nansum(arr))  # Lux·h 累积
    peak = float(np.nanmax(arr))
    return {'DLI_approx': dli, '峰值': peak}


def parse_module_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """解析模块环境数据中的逐小时水温列"""
    df = df.copy()

    col = '逐小时水温℃'
    if col in df.columns:
        parsed = df[col].apply(parse_hourly_string)
        stats = parsed.apply(compute_daily_stats).apply(pd.Series)
        df['水温_日均'] = stats['日均']
        df['水温_std'] = stats['std']
        df['水温_日较差'] = stats['日较差']
        df = df.drop(columns=[col])

    return df


def parse_greenhouse_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """解析温室环境数据中的逐小时气温、湿度、光照列"""
    df = df.copy()

    # 气温
    col = '逐小时气温℃'
    if col in df.columns:
        parsed = df[col].apply(parse_hourly_string)
        stats = parsed.apply(compute_daily_stats).apply(pd.Series)
        df['气温_日均'] = stats['日均']
        df['气温_std'] = stats['std']
        df['气温_日较差'] = stats['日较差']
        df = df.drop(columns=[col])

    # 湿度
    col = '逐小时湿度%'
    if col in df.columns:
        parsed = df[col].apply(parse_hourly_string)
        stats = parsed.apply(compute_daily_stats).apply(pd.Series)
        df['湿度_日均'] = stats['日均']
        df['湿度_std'] = stats['std']
        df = df.drop(columns=[col])

    # 光照
    col = '逐小时光照强度Lux'
    if col in df.columns:
        parsed = df[col].apply(parse_hourly_string)
        stats = parsed.apply(compute_light_stats).apply(pd.Series)
        df['DLI_approx'] = stats['DLI_approx']
        df['光照_峰值'] = stats['峰值']
        df = df.drop(columns=[col])

    return df


def parse_all_hourly(data: dict) -> dict:
    """对所有数据执行逐小时解析"""
    print('\n📦 步骤 2：解析逐小时字符串数据')
    for base in ['红光', '喀左']:
        key_mod = f'{base}_模块环境'
        key_gh = f'{base}_温室环境'

        print(f'  🔄 解析 {key_mod} 逐小时水温...')
        data[key_mod] = parse_module_hourly(data[key_mod])
        print(f'    → 新增列: 水温_日均, 水温_std, 水温_日较差')

        print(f'  🔄 解析 {key_gh} 逐小时气温/湿度/光照...')
        data[key_gh] = parse_greenhouse_hourly(data[key_gh])
        print(f'    → 新增列: 气温_日均, 气温_std, 气温_日较差, 湿度_日均, 湿度_std, DLI_approx, 光照_峰值')

    return data


if __name__ == '__main__':
    from _01_load_and_fix import load_all
    data = load_all()
    data = parse_all_hourly(data)
    for k, v in data.items():
        print(f'  {k}: shape={v.shape}, cols={list(v.columns)}')
