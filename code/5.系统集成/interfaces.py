from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class SensorData:
    """实时传感器数据"""
    timestamp: datetime
    # 环境因子
    water_temp: float      # 水温 (℃)
    air_temp: float        # 气温 (℃) 
    base_do: float         # 溶氧 (mg/L)
    light_lux: float       # 光照 (Lux) - 可能是光照强度或时长
    light_hours: float     # 光照时长 (h) - 如果有的话
    ph: float              # pH值
    ec: float              # EC值 (ms/cm)
    ammonia: float         # 氨氮 (mg/L) - 可能来自人工检测或预测
    # 原始特征行（用于直接喂给已训练模型，避免在线重构特征带来的偏差）
    raw_features: Dict[str, Any] = field(default_factory=dict)
    
    # 原始数据可能包含更多字段，这里列出核心决策字段

@dataclass
class PhysicsState:
    """PINN 反演的物理状态"""
    kla: float             # 复氧系数 (day^-1)
    r_fish: float          # 鱼类耗氧率 (mg/L/d)
    r_bio: float           # 微生物耗氧率 (mg/L/d)
    p_photo: float         # 光合产氧量 (mg/L/d)
    do_deficit: float      # 溶氧亏损量 (mg/L)

@dataclass
class PredictionResult:
    """Chronos 时序预测结果"""
    target_dates: List[datetime]
    water_temp_pred: List[float]
    do_pred: List[float]
    ammonia_pred: List[float] # 稀疏数据补全

@dataclass
class RiskAssessment:
    """XGBoost 病害风险评估"""
    fish_death_prob: float      # 鱼类死亡概率 (0-1)
    veg_disease_prob: float     # 蔬菜病害概率 (0-1)
    risk_level: str             # "Low", "Medium", "High"
    top_risk_factors: List[str] # SHAP 贡献度最高的因子 (如 "High Temp", "Low DO")

@dataclass
class ControlAction:
    """智能控制指令"""
    timestamp: datetime
    aerator_status: bool        # 增氧机开关
    aerator_duration: float     # 建议开启时长 (h)
    feed_amount: float          # 投喂量 (kg/preset) - 只有在鱼类死亡风险高时减少
    light_status: bool          # 补光灯/遮阳网
    reason: str                 # 决策依据 (如 "Predicted DO < 5mg/L & High Fish Respiration")
