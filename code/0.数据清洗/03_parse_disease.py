"""
步骤3：发病文本 → 结构化标签
使用正则表达式解析 内容描述 字段
"""

import pandas as pd
import re


# ──────────────────────── 蔬菜发病解析 ──────────────────────────

def parse_vegetable_record(text: str) -> dict:
    """
    解析蔬菜生长状况文本:
    "生长阶段：正常，生长进度：发芽期，...，预计成熟天：22，预计采收结束天：28，
     病害情况：发现病虫害：叶片发黄"
    """
    result = {
        '生长阶段': None,
        '生长进度': None,
        '预计成熟天': None,
        '有无病虫害': 0,
        '病害类型': None,
    }
    if not isinstance(text, str):
        return result

    m = re.search(r'生长阶段：(\S+?)，', text)
    if m:
        result['生长阶段'] = m.group(1)

    m = re.search(r'生长进度：(\S+?)，', text)
    if m:
        result['生长进度'] = m.group(1)

    m = re.search(r'预计成熟天：(\d+)', text)
    if m:
        result['预计成熟天'] = int(m.group(1))

    if '无病虫害' in text:
        result['有无病虫害'] = 0
    elif '发现病虫害' in text or '病害' in text:
        result['有无病虫害'] = 1
        m = re.search(r'发现病虫害：(.+?)$', text)
        if m:
            result['病害类型'] = m.group(1).strip()

    return result


def parse_vegetable_df(df: pd.DataFrame) -> pd.DataFrame:
    """对蔬菜发病 DataFrame 批量解析"""
    df = df.copy()
    parsed = df['内容描述'].apply(parse_vegetable_record).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    return df


# ──────────────────────── 鱼类发病解析 ──────────────────────────

def parse_fish_record(text: str) -> dict:
    """
    解析鱼类水产状况文本:
    "重量3.45kg，数量3，游动现象：漂浮 翻肚"
    "重量7.4kg，数量6，发现病害：腮丝发白，分叉，烂身，游动现象：漂浮 翻肚"
    """
    result = {
        '死亡重量_kg': None,
        '死亡数量': None,
        '游动异常': None,
        '有无病害': 0,
        '鱼病害类型': None,
    }
    if not isinstance(text, str):
        return result

    m = re.search(r'重量([\d.]+)kg', text)
    if m:
        result['死亡重量_kg'] = float(m.group(1))

    m = re.search(r'数量(\d+)', text)
    if m:
        result['死亡数量'] = int(m.group(1))

    m = re.search(r'游动现象：(.+?)$', text)
    if m:
        result['游动异常'] = m.group(1).strip()

    if '发现病害' in text:
        result['有无病害'] = 1
        m = re.search(r'发现病害：(.+?)，游动', text)
        if m:
            result['鱼病害类型'] = m.group(1).strip()

    return result


def parse_fish_df(df: pd.DataFrame) -> pd.DataFrame:
    """对鱼类发病 DataFrame 批量解析"""
    df = df.copy()
    parsed = df['内容描述'].apply(parse_fish_record).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    return df


# ──────────────────────── 聚合为日级数据 ──────────────────────────

def aggregate_vegetable_daily(df: pd.DataFrame) -> pd.DataFrame:
    """将蔬菜事件聚合为 (日期, 模块) 级别的日统计"""
    df = df.copy()
    df['事件日期'] = pd.to_datetime(df['事件日期'])

    agg = df.groupby(['事件日期', '基地', '模块']).agg(
        蔬菜_事件数=('事件编号', 'count'),
        蔬菜_病害次数=('有无病虫害', 'sum'),
    ).reset_index()

    agg = agg.rename(columns={'事件日期': '日期'})
    return agg


def aggregate_fish_daily(df: pd.DataFrame) -> pd.DataFrame:
    """将鱼类事件聚合为 (日期, 模块) 级别的日统计"""
    df = df.copy()
    df['事件日期'] = pd.to_datetime(df['事件日期'])

    agg = df.groupby(['事件日期', '基地', '模块']).agg(
        鱼_事件数=('事件编号', 'count'),
        鱼_死亡数量=('死亡数量', 'sum'),
        鱼_死亡重量_kg=('死亡重量_kg', 'sum'),
        鱼_病害次数=('有无病害', 'sum'),
    ).reset_index()

    agg = agg.rename(columns={'事件日期': '日期'})
    return agg


def parse_all_disease(data: dict) -> dict:
    """解析全部发病数据并聚合"""
    print('\n📦 步骤 3：发病文本解析与日级聚合')
    for base in ['红光', '喀左']:
        # 蔬菜
        key = f'{base}_蔬菜发病'
        print(f'  🔄 解析 {key} 文本...')
        data[key] = parse_vegetable_df(data[key])
        n_disease = data[key]['有无病虫害'].sum()
        n_total = len(data[key])
        print(f'    → 解析完成: {n_disease}/{n_total} 条含病虫害 ({n_disease/n_total*100:.1f}%)')

        # 聚合
        agg_key = f'{base}_蔬菜日聚合'
        data[agg_key] = aggregate_vegetable_daily(data[key])
        print(f'    → 日级聚合: {len(data[agg_key])} 条 (日期×模块)')

        # 鱼类
        key = f'{base}_鱼类发病'
        print(f'  🔄 解析 {key} 文本...')
        data[key] = parse_fish_df(data[key])
        n_disease = data[key]['有无病害'].sum()
        n_total = len(data[key])
        print(f'    → 解析完成: {n_disease}/{n_total} 条含病害 ({n_disease/n_total*100:.1f}%)')

        agg_key = f'{base}_鱼类日聚合'
        data[agg_key] = aggregate_fish_daily(data[key])
        print(f'    → 日级聚合: {len(data[agg_key])} 条 (日期×模块)')

    return data


if __name__ == '__main__':
    from _01_load_and_fix import load_all
    data = load_all()
    data = parse_all_disease(data)
