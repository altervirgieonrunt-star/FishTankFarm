import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root and script dir to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

# Import interfaces
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))
try:
    from interfaces import SensorData, PhysicsState, RiskAssessment
except ImportError:
    # Dummy for testing
    from dataclasses import dataclass
    @dataclass
    class SensorData:
        timestamp: datetime
        water_temp: float
        base_do: float
        air_temp: float
        light_hours: float
        ammonia: float
        ph: float
        ec: float
        raw_features: dict = None

    class PhysicsState:
        pass
    class RiskAssessment:
        pass

# Try to import xgboost
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("Warning: XGBoost not installed. Using fallback risk rules.")


class DiseaseRiskInference:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = SCRIPT_DIR / "output_v1.2"
        else:
            model_dir = Path(model_dir)

        self.models = {}
        self.feature_names = None
        if xgb:
            for task in ["蔬菜病害", "鱼类死亡"]:
                path = model_dir / f"xgb_{task}_v1.2.json"
                if path.exists():
                    try:
                        clf = xgb.Booster()
                        clf.load_model(str(path))
                        self.models[task] = clf
                        # Record feature names from model (should be identical across tasks)
                        if self.feature_names is None:
                            self.feature_names = clf.feature_names
                        print(f"✅ 加载模型: {task}")
                    except Exception as e:
                        print(f"❌ 模型加载失败 {task}: {e}")
                else:
                    print(f"⚠️ 未找到模型: {path}")
        else:
            print("⚠️ 使用规则回退（无 XGBoost）")
        self.using_xgb_models = bool(self.models)

        # Buffer for temporal features (store last 7+ days of data)
        self.buffer = []
        self.max_buffer_days = 8

        # Minimal online features as fallback (when raw_features are not available)
        self.fallback_feature_cols = [
            "水温_日均", "溶氧mg/L", "光照时长h", "气温_日均", "PH", "EC值ms/cm", "氨氮mg/L",
            "pinn_DO_deficit", "pinn_R_fish_T", "pinn_reaeration", "pinn_oxygen_stress", "pinn_P_photo",
            "trend_水温_3d", "DO_volatility", "水气温_交互", "高温天数_7d", "低氧天数_7d",
        ]

    def update(self, data: SensorData):
        """Update buffer with new data point"""
        self.buffer.append(data)
        cutoff = data.timestamp - timedelta(days=self.max_buffer_days)
        self.buffer = [d for d in self.buffer if d.timestamp > cutoff]

    def _compute_rolling_stats(self):
        """Compute rolling means/stats from buffer to mimic daily aggregated data"""
        if not self.buffer:
            return {}

        df = pd.DataFrame([vars(d) for d in self.buffer])
        df.set_index("timestamp", inplace=True)

        now = self.buffer[-1].timestamp
        last_24h = df[df.index > now - timedelta(hours=24)]
        last_3d = df[df.index > now - timedelta(days=3)]
        last_7d = df[df.index > now - timedelta(days=7)]

        stats = {}
        if len(last_24h) > 0:
            stats["水温_日均"] = last_24h["water_temp"].mean()
            stats["溶氧mg/L"] = last_24h["base_do"].mean()
            stats["气温_日均"] = last_24h["air_temp"].mean()
            stats["PH"] = last_24h["ph"].mean()
            stats["EC值ms/cm"] = last_24h["ec"].mean()
            stats["氨氮mg/L"] = last_24h["ammonia"].mean()
            stats["光照时长h"] = last_24h["light_hours"].max()
        else:
            last = self.buffer[-1]
            stats["水温_日均"] = last.water_temp
            stats["溶氧mg/L"] = last.base_do
            stats["气温_日均"] = last.air_temp
            stats["PH"] = last.ph
            stats["EC值ms/cm"] = last.ec
            stats["氨氮mg/L"] = last.ammonia
            stats["光照时长h"] = last.light_hours

        t_mean_3d = last_3d["water_temp"].mean() if len(last_3d) > 0 else stats["水温_日均"]
        stats["trend_水温_3d"] = stats["水温_日均"] - t_mean_3d

        if len(last_24h) > 1:
            stats["DO_volatility"] = last_24h["base_do"].diff().abs().mean()
        else:
            stats["DO_volatility"] = 0.0

        stats["水气温_交互"] = stats["水温_日均"] * stats["气温_日均"] / 100.0

        if len(last_7d) > 0:
            high_temp_ratio = (last_7d["water_temp"] > 28).mean()
            stats["高温天数_7d"] = high_temp_ratio * 7.0
            low_do_ratio = (last_7d["base_do"] < 5).mean()
            stats["低氧天数_7d"] = low_do_ratio * 7.0
        else:
            stats["高温天数_7d"] = 0.0
            stats["低氧天数_7d"] = 0.0

        return stats

    def _build_feature_vector(self, data: SensorData, phys_state: PhysicsState):
        """
        Prefer raw feature row (from featured_*.csv) to align with training.
        Fallback to lightweight online features if raw features are missing.
        """
        if self.feature_names and getattr(data, "raw_features", None):
            feats = {}
            feats.update(data.raw_features)
            # Add PINN features if not already present
            feats.setdefault("pinn_DO_deficit", phys_state.do_deficit)
            feats.setdefault("pinn_reaeration", phys_state.kla * phys_state.do_deficit)
            feats.setdefault("pinn_R_fish_T", phys_state.r_fish)
            feats.setdefault("pinn_P_photo", phys_state.p_photo)
            feats.setdefault(
                "pinn_oxygen_stress",
                phys_state.r_fish + phys_state.r_bio - feats.get("pinn_reaeration", 0.0) - phys_state.p_photo,
            )
            # Align to model feature order
            vector = [feats.get(col, 0.0) for col in self.feature_names]
            return vector, "raw"

        # Fallback path
        feats = self._compute_rolling_stats()
        feats["pinn_DO_deficit"] = phys_state.do_deficit
        feats["pinn_reaeration"] = phys_state.kla * feats["pinn_DO_deficit"]
        feats["pinn_R_fish_T"] = phys_state.r_fish
        feats["pinn_P_photo"] = phys_state.p_photo
        feats["pinn_oxygen_stress"] = (
            phys_state.r_fish + phys_state.r_bio - feats["pinn_reaeration"] - phys_state.p_photo
        )
        vector = [feats.get(col, 0.0) for col in self.fallback_feature_cols]
        return vector, "fallback"

    def predict(self, phys_state: PhysicsState) -> RiskAssessment:
        if not self.buffer:
            return None

        results = {}
        top_factors = []

        if xgb and self.models:
            try:
                input_vector, mode = self._build_feature_vector(self.buffer[-1], phys_state)
                feature_names = self.feature_names if mode == "raw" else self.fallback_feature_cols
                dtest = xgb.DMatrix([input_vector], feature_names=feature_names)
                for task, model in self.models.items():
                    prob = model.predict(dtest)[0]
                    results[task] = float(prob)
            except Exception as e:
                print(f"Prediction failed: {e}")
        if (not results) and (not (xgb and self.models)):
            feats = self._compute_rolling_stats()
            if feats.get("高温天数_7d", 0) > 3:
                results["鱼类死亡"] = 0.8
                results["蔬菜病害"] = 0.7
            elif feats.get("溶氧mg/L", 6) < 4:
                results["鱼类死亡"] = 0.6
                results["蔬菜病害"] = 0.3
            else:
                results["鱼类死亡"] = 0.05
                results["蔬菜病害"] = 0.1
        elif not results:
            feats = self._compute_rolling_stats()
            # 模型推理异常时，使用同一套规则回退，避免静默归零
            if feats.get("高温天数_7d", 0) > 3:
                results["鱼类死亡"] = 0.8
                results["蔬菜病害"] = 0.7
            elif feats.get("溶氧mg/L", 6) < 4:
                results["鱼类死亡"] = 0.6
                results["蔬菜病害"] = 0.3
            else:
                results["鱼类死亡"] = 0.05
                results["蔬菜病害"] = 0.1

        fish_prob = results.get("鱼类死亡", 0.0)
        veg_prob = results.get("蔬菜病害", 0.0)

        risk_level = "Low"
        if fish_prob > 0.5 or veg_prob > 0.6:
            risk_level = "High"
        elif fish_prob > 0.3 or veg_prob > 0.4:
            risk_level = "Medium"

        # Simple factor hints (fallback)
        feats = self._compute_rolling_stats()
        if feats.get("高温天数_7d", 0) > 0:
            top_factors.append("High Temp Duration")
        if feats.get("溶氧mg/L", 6) < 4:
            top_factors.append("Low DO")
        if feats.get("光照时长h", 0) < 6:
            top_factors.append("Low Light")

        return RiskAssessment(
            fish_death_prob=fish_prob,
            veg_disease_prob=veg_prob,
            risk_level=risk_level,
            top_risk_factors=top_factors,
        )


if __name__ == "__main__":
    inf = DiseaseRiskInference()
    print("Disease Inference initialized")
