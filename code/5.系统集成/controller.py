import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import interfaces
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))
try:
    from interfaces import SensorData, PhysicsState, RiskAssessment, PredictionResult, ControlAction
except ImportError:
    # Fallback if running from dashboard where path might differ or already imported
    from .interfaces import SensorData, PhysicsState, RiskAssessment, PredictionResult, ControlAction

import importlib.util

def load_source(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import Inference Modules via importlib (to avoid name collision 'inference')
chronos_mod = load_source("chronos_inference", PROJECT_ROOT / "code" / "2.时序预测" / "inference.py")
ChronosInference = chronos_mod.ChronosInference

pinn_mod = load_source("pinn_inference", PROJECT_ROOT / "code" / "3.PINN" / "inference.py")
PhysicsInference = pinn_mod.PhysicsInference

disease_mod = load_source("disease_inference", PROJECT_ROOT / "code" / "1.病害预测" / "inference.py")
DiseaseRiskInference = disease_mod.DiseaseRiskInference

class MPCController:
    def __init__(self, site="红光"):
        self.chronos = ChronosInference()
        self.pinn = PhysicsInference(site=site)
        self.disease_model = DiseaseRiskInference()
        self.site = site
        
        # MPC Parameters
        self.horizon = 24 # 24 hours lookahead
        self.risk_threshold = 0.3
        self.energy_cost_per_hour = 0.5 # yuan
        
    def step(self, current_data: SensorData) -> dict:
        """
        Execute one control step:
        1. Update state
        2. Predict future
        3. Optimize action
        """
        # 1. Update internal states
        self.chronos.update(current_data)
        self.disease_model.update(current_data)
        
        # Get current physical state
        phys_state = self.pinn.get_state(current_data)
        
        # Get current risk
        current_risk = self.disease_model.predict(phys_state)
        
        # 2. Predict future (Base Scenario - No Action or Default Action)
        # We assume "Default Action" means keeping current state? 
        # Or Chronos inherently predicts "likely future based on history".
        forecast = self.chronos.predict(horizon=self.horizon)
        
        # 3. Decision Logic (Simplified MPC)
        # We evaluate: "Do nothing" vs "Action A" vs "Action B"
        # However, simulating "Action A" effect on Chronos prediction is hard.
        # So we use a hybrid approach:
        # - Use Chronos for "External Environment" (Temp, Light mostly external)
        # - Use Physics for "Process Variables" (DO, Ammonia) response to Action.
        
        action = self._optimize_action(current_data, current_risk, phys_state, forecast)
        
        return {
            "timestamp": current_data.timestamp,
            "sensor": current_data,
            "physics": phys_state,
            "risk": current_risk,
            "forecast": forecast,
            "action": action
        }

    def _optimize_action(self, current_data, current_risk, phys_state, forecast) -> ControlAction:
        """
        Determine optimal control action.
        Strategy:
        1. If Risk is Low -> Energy Saving Mode (Off)
        2. If Risk is High -> Evaluate Actions to mitigate
        """
        
        # Default: All off
        action = ControlAction(
            timestamp=current_data.timestamp,
            aerator_status=False,
            aerator_duration=0.0,
            feed_amount=10.0, # Normal feed
            light_status=False,
            reason="System Stable"
        )
        
        # 1. Check Oxygen Risk
        # If predicted DO drops below 4.0 OR Current DO < 5.0
        min_pred_do = min(forecast.do_pred) if forecast.do_pred else current_data.base_do
        
        if current_data.base_do < 5.0 or min_pred_do < 4.5:
            # Need Aeration
            # Use PINN状态估计动态给出增氧时长（替代固定2小时）
            target_do = 6.0
            severity = max(0.0, target_do - min(current_data.base_do, min_pred_do))
            # 近似复氧速率：与 K_La 和 DO deficit 成正相关，设置下限避免除零
            est_recovery_rate = max(0.2, phys_state.kla * max(0.1, phys_state.do_deficit) / 8.0)
            est_duration = severity / est_recovery_rate

            action.aerator_status = True
            action.aerator_duration = float(np.clip(est_duration, 0.5, 6.0))
            action.reason = f"Low DO Detected (Current: {current_data.base_do:.1f}, Pred Min: {min_pred_do:.1f})"
            
            # If extremely low, reduce feed
            if current_data.base_do < 3.0:
                action.feed_amount = 0.0
                action.reason += " | Critical DO! Stop Feeding."
                
        # 2. Check Disease Risk (High Temp / Low Light)
        if current_risk.risk_level in ["Medium", "High"]:
             # Causal finding: Light -> Disease (Negative correlation in Hongguang)
             # So increasing light reduces disease.
             if "Low Light" in current_risk.top_risk_factors or current_data.light_hours < 8.0:
                 action.light_status = True
                 action.reason = f"High Disease Risk ({current_risk.fish_death_prob:.2f}). Supplement Light."
                 
        return action
