import sys
from pathlib import Path
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List

# 添加项目根目录到 sys.path 以便导入 interfaces
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import interfaces via sys.path
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))
try:
    from interfaces import SensorData, PredictionResult
except ImportError:
    # Fallback if module name is different due to chinese characters
    # Creating dummy classes for local testing if import fails
    from dataclasses import dataclass
    @dataclass
    class SensorData:
        timestamp: datetime
        water_temp: float
        base_do: float
        ammonia: float
        
    @dataclass
    class PredictionResult:
        target_dates: List[datetime]
        water_temp_pred: List[float]
        do_pred: List[float]
        ammonia_pred: List[float]

# 尝试导入 Chronos
try:
    from chronos import ChronosPipeline
except ImportError:
    print("Warning: Chronos not installed. Using data-driven fallback forecaster.")
    ChronosPipeline = None

class ChronosInference:
    def __init__(self, model_path: str = None, device: str = None, context_len: int = 48):
        self.context_len = context_len
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if model_path is None:
            model_path = str(SCRIPT_DIR / "models" / "chronos-t5-tiny")
            
        self.model = None
        if ChronosPipeline:
            try:
                self.model = ChronosPipeline.from_pretrained(
                    model_path,
                    device_map=self.device,
                    torch_dtype=torch.float32,
                )
                print(f"✅ Chronos 模型已加载至 {self.device}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
        self.is_real_model = self.model is not None
        
        # 数据缓冲区: {variable_name: [values...]}
        self.buffer = {
            "water_temp": [],
            "do": [],
            "ammonia": [] # 即使稀疏也尝试缓冲
        }
        self.timestamps = []
        
    def update(self, data: SensorData):
        """更新内部缓冲区"""
        self.timestamps.append(data.timestamp)
        self.buffer["water_temp"].append(data.water_temp)
        self.buffer["do"].append(data.base_do)
        # 对于氨氮，如果是0 (缺失)，我们可能需要前向填充或插值
        # 这里简单处理：如果有值则更新，无值则沿用上一次 (Forward Fill)
        # 但 SensorData 传入的是 floats, 0.0 可能代表缺失
        val_ammonia = data.ammonia
        if val_ammonia == 0.0 and len(self.buffer["ammonia"]) > 0:
            val_ammonia = self.buffer["ammonia"][-1]
        self.buffer["ammonia"].append(val_ammonia)
        
        # 保持缓冲区长度
        if len(self.timestamps) > self.context_len * 2: # 稍微保留多一点以备不时之需
            start = -self.context_len
            self.timestamps = self.timestamps[start:]
            for k in self.buffer:
                self.buffer[k] = self.buffer[k][start:]
                
    def predict(self, horizon: int = 24) -> PredictionResult:
        """预测未来 horizon 个时间步 (假设 step 是小时或天，取决于输入频率)"""
        if not self.model:
            # Data-driven fallback when Chronos package/model is unavailable
            return self._fallback_predict(horizon)
            
        preds = {}
        for key in ["water_temp", "do", "ammonia"]:
            series = np.array(self.buffer[key])
            if len(series) < 10: # 数据太少，无法预测
                preds[key] = [series[-1]] * horizon if len(series) > 0 else [0.0] * horizon
                continue
                
            # 准备 context
            context = torch.tensor(series[-self.context_len:], dtype=torch.float32)
            
            # 推理
            try:
                forecast = self.model.predict(
                    context,
                    prediction_length=horizon,
                    num_samples=20, # 快速采样
                )
                # 取中位数
                forecast_np = forecast.numpy().squeeze(0) # (num_samples, horizon)
                median_pred = np.median(forecast_np, axis=0) # (horizon,)
                preds[key] = median_pred.tolist()
            except Exception:
                preds[key] = self._series_fallback(self.buffer[key], horizon, non_negative=(key != "water_temp"))
            
        # 生成对应的时间戳
        last_time = self.timestamps[-1]
        # 优先使用最近两点的真实采样间隔
        step_delta = timedelta(days=1)
        if len(self.timestamps) >= 2:
            inferred = self.timestamps[-1] - self.timestamps[-2]
            if inferred.total_seconds() > 0:
                step_delta = inferred
        target_dates = [last_time + step_delta * (i + 1) for i in range(horizon)]
        
        return PredictionResult(
            target_dates=target_dates,
            water_temp_pred=preds["water_temp"],
            do_pred=preds["do"],
            ammonia_pred=preds["ammonia"]
        )

    def _series_fallback(self, series, horizon: int, non_negative: bool = False) -> List[float]:
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            return [0.0] * horizon
        win = min(7, arr.size)
        recent = arr[-win:]
        trend = float(np.mean(np.diff(recent))) if win > 1 else 0.0
        base = float(recent[-1])
        pred = []
        for i in range(horizon):
            v = base + trend * (i + 1)
            if non_negative:
                v = max(0.0, v)
            pred.append(float(v))
        return pred

    def _fallback_predict(self, horizon):
        """Fallback prediction based on recent sequence trend (no mock waveform)."""
        last_time = self.timestamps[-1] if self.timestamps else datetime.now()
        step_delta = timedelta(days=1)
        if len(self.timestamps) >= 2:
            inferred = self.timestamps[-1] - self.timestamps[-2]
            if inferred.total_seconds() > 0:
                step_delta = inferred
        target_dates = [last_time + step_delta * (i + 1) for i in range(horizon)]
        return PredictionResult(
            target_dates=target_dates,
            water_temp_pred=self._series_fallback(self.buffer["water_temp"], horizon, non_negative=False),
            do_pred=self._series_fallback(self.buffer["do"], horizon, non_negative=True),
            ammonia_pred=self._series_fallback(self.buffer["ammonia"], horizon, non_negative=True),
        )

if __name__ == "__main__":
    # Test
    inf = ChronosInference()
    print("Inference initialized")
