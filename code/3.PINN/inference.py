import sys
import torch
import numpy as np
from pathlib import Path

# Add project root and script dir to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# Import from project modules
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))
try:
    from interfaces import SensorData, PhysicsState
except ImportError:
    # Fallback for local testing
    from dataclasses import dataclass
    @dataclass
    class SensorData:
        water_temp: float
        base_do: float
        light_hours: float
    @dataclass
    class PhysicsState:
        kla: float
        r_fish: float
        r_bio: float
        p_photo: float
        do_deficit: float

from physics import PhysicsParams, do_saturation
from config import PHYSICS_PARAMS, PARAM_BOUNDS

class PhysicsInference:
    def __init__(self, site: str = "红光", model_dir: str = None):
        self.site = site
        self.device = "cpu" # Inference on CPU is fine and safer
        
        if model_dir is None:
            model_dir = SCRIPT_DIR / "output"
        else:
            model_dir = Path(model_dir)
            
        model_path = model_dir / f"pinn_{site}.pt"
        
        # Initialize PhysicsParams structure
        self.physics = PhysicsParams(PHYSICS_PARAMS, PARAM_BOUNDS).to(self.device)
        
        # Load learned parameters
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.physics.load_state_dict(checkpoint["physics_state"])
                print(f"✅ PINN 物理参数已加载: {model_path}")
            except Exception as e:
                print(f"❌ PINN 参数加载失败: {e}")
        else:
            print(f"⚠️ 未找到 PINN 模型文件: {model_path}，使用初始参数")
            
        self.physics.eval()
        
    def get_state(self, data: SensorData) -> PhysicsState:
        """
        根据当前传感器数据和学习到的物理参数，计算系统物理状态
        """
        # 1. 获取物理参数 (Scalar)
        K_La = self.physics.get("K_La").item()
        R_fish_base = self.physics.get("R_fish_base").item()
        alpha_T = self.physics.get("alpha_T").item()
        R_bio = self.physics.get("R_bio").item()
        P_photo_rate = self.physics.get("P_photo_rate").item()
        
        # 2. 计算动态变量
        T = data.water_temp
        DO = data.base_do
        light_h = data.light_hours if hasattr(data, "light_hours") else 0.0
        
        # 饱和溶氧
        # do_saturation expects Tensor, wrapper for float
        do_sat_val = do_saturation(torch.tensor(float(T))).item()
        
        # 溶氧亏损
        do_deficit = do_sat_val - DO
        
        # 鱼类耗氧 (温度修正)
        # R_fish(T) = R_fish_base * (1 + alpha_T * (T - 25))
        r_fish_T = R_fish_base * (1.0 + alpha_T * (T - 25.0))
        # 保证不为负 (虽然物理上 R_fish > 0, 但 alpha_T 极大时可能异常，这里做个保护)
        r_fish_T = max(0.0, r_fish_T)
        
        # 光合产氧
        p_photo = P_photo_rate * light_h
        
        return PhysicsState(
            kla=K_La,
            r_fish=r_fish_T,
            r_bio=R_bio,
            p_photo=p_photo,
            do_deficit=do_deficit
        )

if __name__ == "__main__":
    # Test
    inf = PhysicsInference(site="红光")
    dummy_data = SensorData(water_temp=25.0, base_do=6.0, light_hours=12.0)
    state = inf.get_state(dummy_data)
    print(state)
