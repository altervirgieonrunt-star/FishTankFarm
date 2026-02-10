"""
步骤1：加载 Excel 数据并修复格式问题
- 喀左鱼类发病数据：补表头
- 日期列统一转 pd.Timestamp
- 丢弃全空列（弧菌、余氯、盐度、离子、经纬度）
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '资料包', '训练数据')

# 喀左鱼类发病数据缺表头，手动指定列名（与红光鱼类一致，但少经纬度2列）
KAZUO_FISH_COLUMNS = [
    '事件编号', '种养记录编号', '标准名称', '基地', '温室',
    '模块', '单元', '事件类型', '内容描述', '事件时间'
]

# 模块环境数据中完全空白的列（缺失率≈100%）
MODULE_DROP_COLS = ['弧菌CFU/ml', '余氯mg/L', '盐度‰', '钾离子mg/L', '钠离子mg/L', '镁离子mg/L']

# 发病数据中完全空白的列
EVENT_DROP_COLS = ['经度', '纬度']


def load_module_env(base: str) -> pd.DataFrame:
    """加载模块环境日数据"""
    fname = f'{base}模块环境日数据.xlsx'
    df = pd.read_excel(os.path.join(DATA_DIR, fname))
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

    # 丢弃全空列
    cols_to_drop = [c for c in MODULE_DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    print(f'  ✅ {fname}: {len(df)} 行, {len(df.columns)} 列 (已丢弃 {len(cols_to_drop)} 个全空列)')
    return df


def load_greenhouse_env(base: str) -> pd.DataFrame:
    """加载温室环境日数据"""
    fname = f'{base}温室环境日数据.xlsx'
    df = pd.read_excel(os.path.join(DATA_DIR, fname))
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')

    print(f'  ✅ {fname}: {len(df)} 行, {len(df.columns)} 列')
    return df


def load_disease_data(base: str, category: str) -> pd.DataFrame:
    """
    加载发病数据
    base: '红光' | '喀左'
    category: '蔬菜' | '鱼类'
    """
    fname = f'{base}{category}发病数据.xlsx'
    fpath = os.path.join(DATA_DIR, fname)

    if base == '喀左' and category == '鱼类':
        # 该文件缺少表头，首行即为数据
        df = pd.read_excel(fpath, header=None)
        df.columns = KAZUO_FISH_COLUMNS
    else:
        df = pd.read_excel(fpath)
        # 丢弃空的经纬度列
        cols_to_drop = [c for c in EVENT_DROP_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    # 统一事件时间为 Timestamp
    df['事件时间'] = pd.to_datetime(df['事件时间'], errors='coerce')
    # 提取日期（不含时间）用于后续聚合
    df['事件日期'] = df['事件时间'].dt.date

    print(f'  ✅ {fname}: {len(df)} 行, {len(df.columns)} 列'
          + (' (已手动补表头)' if base == '喀左' and category == '鱼类' else ''))
    return df


def load_all():
    """加载全部 8 个文件，返回字典"""
    print('📦 步骤 1：加载与修复格式')
    data = {}
    for base in ['红光', '喀左']:
        print(f'\n  📍 {base}基地:')
        data[f'{base}_模块环境'] = load_module_env(base)
        data[f'{base}_温室环境'] = load_greenhouse_env(base)
        data[f'{base}_蔬菜发病'] = load_disease_data(base, '蔬菜')
        data[f'{base}_鱼类发病'] = load_disease_data(base, '鱼类')
    return data


if __name__ == '__main__':
    data = load_all()
    for k, v in data.items():
        print(f'  {k}: shape={v.shape}')
