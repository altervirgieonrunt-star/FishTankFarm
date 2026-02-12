import pandas as pd
from typing import Generator, Dict, Any
from pathlib import Path
try:
    from interfaces import SensorData
except ImportError:
    from .interfaces import SensorData

class DataLoader:
    def __init__(self, data_path: str, site_name: str = "红光"):
        self.data_path = Path(data_path)
        self.site_name = site_name
        self.df = self._load_data()
        self.current_index = 0
        
    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # 尝试解析日期索引
        # 假设第一列是日期，或者有名为 'date', 'time', 'timestamp' 的列
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or '日期' in col:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        else:
            raise ValueError("未找到日期列（需包含 date/time/timestamp/日期）")
            
        # 1. 找到溶氧列名
        do_col = None
        for col in ['溶氧mg/L', '溶氧', 'DO']:
            if col in df.columns:
                do_col = col
                break
        
        # 2. 如果找到溶氧列，先进行 ffill (处理中间缺失)，然后 dropna (处理开头缺失)
        if do_col:
            # Forward fill to handle small gaps in sensor data
            df = df.ffill()
            # Drop rows at the start where DO is still NaN (because ffill doesn't fix start)
            df = df.dropna(subset=[do_col])
            
        self.date_col = date_col
        return df

    def stream(self) -> Generator[SensorData, None, None]:
        """模拟实时数据流，逐行 yield SensorData"""
        for _, row in self.df.iterrows():
            yield self._row_to_sensor_data(row)

    def _row_to_sensor_data(self, row: pd.Series) -> SensorData:
        # 映射列名 (根据 feature_红光.csv 的常见列名)
        # 注意：这里需要根据实际 CSV 列名进行调整，这里使用一种鲁棒的查找方式
        
        timestamp = row[self.date_col]
        raw_features = self._build_raw_features(row)
        
        return SensorData(
            timestamp=timestamp,
            water_temp=self._get_val(row, ['水温_日均', '水温', 'Water_Temp']),
            air_temp=self._get_val(row, ['气温_日均', '气温', 'Air_Temp']),
            base_do=self._get_val(row, ['溶氧mg/L', '溶氧', 'DO']),
            light_lux=self._get_val(row, ['光照_峰值', '光照', 'Light_Peak']),
            light_hours=self._get_val(row, ['光照时长h', '光照时长', 'Light_Hours']),
            ph=self._get_val(row, ['PH', 'pH']),
            ec=self._get_val(row, ['EC值ms/cm', 'EC', 'EC_val']),
            ammonia=self._get_val(row, ['氨氮mg/L', '氨氮', 'Ammonia'], default=0.0),
            raw_features=raw_features,
        )
    
    def _get_val(self, row, candidates, default=0.0):
        for col in candidates:
            if col in row.index:
                return self._to_float(row[col], default=default)
        return default

    def _build_raw_features(self, row: pd.Series) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}
        for col, val in row.items():
            if col == self.date_col:
                continue
            if pd.api.types.is_number(val) and not pd.isna(val):
                raw[col] = float(val)
            else:
                raw[col] = val
        return raw

    @staticmethod
    def _to_float(val, default=0.0):
        if pd.isna(val):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append("..") # 为了能找到 data_path
    
    # 假设路径
    csv_path = "../../data/featured_红光.csv" 
    loader = DataLoader(csv_path)
    
    print("开始模拟数据流...")
    for i, data in enumerate(loader.stream()):
        if i >= 5: break
        print(f"[{i}] {data}")
