"""
实时预处理器 (Processor Class)
将离线流水线的核心逻辑封装为可复用的类，支持:
  - 单条 dict 输入
  - 单行 DataFrame 输入
  - 批量 DataFrame 输入
  - 决赛现场 predict() 内部调用

解决审查反馈: "缺失预测态集成" + "错误处理机制"

使用示例:
    from processor import AquaponicsProcessor
    proc = AquaponicsProcessor()
    clean_row = proc.process({"水温_日均": 25.5, "溶氧mg/L": 8.2, ...})
"""

import pandas as pd
import numpy as np
import warnings
import traceback

# ──────── 领域知识配置 ────────

HARD_BOUNDS = {
    '最低水温℃':   (0, 45),    '最高水温℃':   (0, 45),
    '水温_日均':    (0, 45),    '水温_std':     (0, 20),
    '最低气温℃':   (-30, 55),  '最高气温℃':   (-30, 55),
    '气温_日均':    (-30, 55),  '湿度_日均':    (0, 100),
    '最低湿度%':    (0, 100),   '最高湿度%':    (0, 100),
    '溶氧mg/L':    (0, 25),    '氨氮mg/L':    (0, 50),
    'PH':          (3, 11),    'EC值ms/cm':   (0, 20),
    '能耗km/h':    (0, 500),   '光照时长h':   (0, 24),
}

# 标准列名（期望输入包含这些列的子集）
EXPECTED_COLUMNS = [
    '日期', '基地', '温室', '模块',
    '最低水温℃', '最高水温℃', '水温_日均',
    '最低气温℃', '最高气温℃', '气温_日均',
    '最低湿度%', '最高湿度%', '湿度_日均',
    '溶氧mg/L', '氨氮mg/L', '亚盐mg/L', 'PH', 'EC值ms/cm',
    '能耗km/h', '光照时长h',
]

# 列名别名映射（处理可能的不同命名风格）
COLUMN_ALIASES = {
    '水温': '水温_日均',
    'water_temp': '水温_日均',
    'DO': '溶氧mg/L',
    'do': '溶氧mg/L',
    'NH3': '氨氮mg/L',
    'nh3': '氨氮mg/L',
    'ph': 'PH',
    'pH': 'PH',
    'ec': 'EC值ms/cm',
    'EC': 'EC值ms/cm',
    'temp': '气温_日均',
    'humidity': '湿度_日均',
}


def calc_do_saturation(water_temp: float) -> float:
    """基于水温计算理论溶氧饱和度 (mg/L)"""
    T = max(0, min(45, water_temp))
    return max(0, 14.62 - 0.3898 * T + 0.006969 * T**2 - 0.00005896 * T**3)


class AquaponicsProcessor:
    """
    鱼菜共生数据实时预处理器

    封装了:
    1. 列名标准化与别名映射
    2. 硬边界异常值剔除
    3. 物理特征计算 (DO饱和度, 温差耦合)
    4. 有效性标记
    5. 全程 try-except 保底 (绝不崩溃)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._log_buffer = []

    def _log(self, msg: str):
        self._log_buffer.append(msg)
        if self.verbose:
            print(f'  [Processor] {msg}')

    def get_logs(self) -> list[str]:
        """获取处理日志"""
        return list(self._log_buffer)

    def clear_logs(self):
        self._log_buffer.clear()

    # ────────────── 核心处理流程 ──────────────

    def process(self, data) -> pd.DataFrame:
        """
        主入口：接受多种输入格式，返回清洗后的 DataFrame

        参数:
            data: dict | pd.Series | pd.DataFrame | list[dict]

        返回:
            pd.DataFrame: 清洗后的数据（即使出错也返回尽可能可用的结果）
        """
        self.clear_logs()

        try:
            df = self._normalize_input(data)
            df = self._standardize_columns(df)
            df = self._enforce_hard_bounds(df)
            df = self._add_validity_flags(df)
            df = self._add_physics_features(df)
            self._log(f'✅ 处理完成: {len(df)} 行, {len(df.columns)} 列')
        except Exception as e:
            self._log(f'❌ 处理异常: {e}')
            self._log(traceback.format_exc())
            # 尽力返回原始数据
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame()
            warnings.warn(f'AquaponicsProcessor 异常，返回原始数据: {e}')

        return df

    def _normalize_input(self, data) -> pd.DataFrame:
        """将各种输入格式统一为 DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, pd.Series):
            return pd.DataFrame([data])
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise TypeError(f'不支持的输入类型: {type(data)}')

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """列名标准化：去空格、应用别名映射"""
        # 去除前后空格
        df.columns = [c.strip() for c in df.columns]

        # 应用别名
        rename_map = {}
        for col in df.columns:
            if col in COLUMN_ALIASES and COLUMN_ALIASES[col] not in df.columns:
                rename_map[col] = COLUMN_ALIASES[col]

        if rename_map:
            df = df.rename(columns=rename_map)
            self._log(f'列名映射: {rename_map}')

        return df

    def _enforce_hard_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """硬边界截断"""
        for col, (lo, hi) in HARD_BOUNDS.items():
            if col not in df.columns:
                continue
            # 强制转数值
            df[col] = pd.to_numeric(df[col], errors='coerce')
            out_mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            n_out = out_mask.sum()
            if n_out > 0:
                df.loc[out_mask, col] = np.nan
                self._log(f'⚠️ {col}: {n_out} 条越界 → NaN')
        return df

    def _add_validity_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """为高缺失率指标添加有效标记"""
        flag_cols = ['溶氧mg/L', '氨氮mg/L', '亚盐mg/L', 'PH', 'EC值ms/cm']
        for col in flag_cols:
            if col in df.columns:
                flag = col.replace('mg/L', '').replace('ms/cm', '').replace('值', '').rstrip()
                df[f'{flag}_有效'] = df[col].notna().astype(int)
        return df

    def _add_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加物理衍生特征"""
        # DO 饱和度
        if '水温_日均' in df.columns:
            df['DO_饱和度_理论'] = df['水温_日均'].apply(
                lambda t: calc_do_saturation(t) if pd.notna(t) else np.nan
            )
            if '溶氧mg/L' in df.columns:
                df['DO_饱和比'] = df['溶氧mg/L'] / df['DO_饱和度_理论'].replace(0, np.nan)
                df['DO_亏损'] = df['DO_饱和度_理论'] - df['溶氧mg/L']

        # 水气温差
        if '水温_日均' in df.columns and '气温_日均' in df.columns:
            df['水气温差'] = df['水温_日均'] - df['气温_日均']

        return df


# ────────────── 便捷函数 ──────────────

_default_processor = None

def robust_preprocess(data) -> pd.DataFrame:
    """
    便捷函数: 直接调用默认 Processor
    用于在 predict() 内部一行调用:

        from processor import robust_preprocess
        clean_data = robust_preprocess(raw_input)
        prediction = model.predict(clean_data)
    """
    global _default_processor
    if _default_processor is None:
        _default_processor = AquaponicsProcessor(verbose=False)
    return _default_processor.process(data)


if __name__ == '__main__':
    # 测试: 单条 dict 输入
    print('=' * 60)
    print('🧪 Processor 功能测试')
    print('=' * 60)

    proc = AquaponicsProcessor(verbose=True)

    # 测试1: 正常输入
    print('\n── 测试1: 正常 dict 输入 ──')
    result = proc.process({
        '水温_日均': 25.5,
        '溶氧mg/L': 8.2,
        '氨氮mg/L': 0.3,
        'PH': 7.1,
        '气温_日均': 22.0,
        '湿度_日均': 75.0,
    })
    print(f'  输出列: {list(result.columns)}')
    print(result.to_string(index=False))

    # 测试2: 异常值输入
    print('\n── 测试2: 含异常值的输入 ──')
    result = proc.process({
        '水温_日均': 999.0,   # 明显不合理
        '溶氧mg/L': -5.0,    # 负数
        'PH': 2.0,           # 过低
    })
    print(result.to_string(index=False))

    # 测试3: 别名输入
    print('\n── 测试3: 英文别名输入 ──')
    result = proc.process({
        'water_temp': 26.0,
        'DO': 7.5,
        'NH3': 0.5,
        'pH': 7.2,
    })
    print(f'  输出列: {list(result.columns)}')
    print(result.to_string(index=False))

    # 测试4: 空输入
    print('\n── 测试4: 空 dict 输入 ──')
    result = proc.process({})
    print(f'  shape: {result.shape}')

    # 测试5: 批量 DataFrame
    print('\n── 测试5: 批量 DataFrame 输入 ──')
    batch = pd.DataFrame([
        {'水温_日均': 24, '溶氧mg/L': 9.0, '气温_日均': 20},
        {'水温_日均': 30, '溶氧mg/L': 6.0, '气温_日均': 28},
        {'水温_日均': 15, '溶氧mg/L': 12.0, '气温_日均': 10},
    ])
    result = proc.process(batch)
    print(result.to_string(index=False))

    print('\n✅ 全部测试完成!')
