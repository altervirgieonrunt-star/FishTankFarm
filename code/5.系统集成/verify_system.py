import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Add specific directories to sys.path to avoid invalid syntax with numbered dirs
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))
import data_loader
from data_loader import DataLoader
import controller
from controller import MPCController


def main():
    print("=" * 60)
    print("  🔧 系统集成验证 (Headless Verification)")
    print("=" * 60)
    
    # 1. Initialize Loader
    print("\n[1/3] 初始化数据加载器...")
    try:
        data_path = PROJECT_ROOT / "data" / "featured_红光.csv"
        loader = DataLoader(str(data_path), site_name="红光")
        print("  ✅ Data Loader Ready")
    except Exception as e:
        print(f"  ❌ Data Loader Failed: {e}")
        return

    # 2. Initialize Controller
    print("\n[2/3] 初始化控制器 (含 Chronos, PINN, XGBoost)...")
    try:
        controller = MPCController(site="红光")
        print("  ✅ Controller Ready")
    except Exception as e:
        print(f"  ❌ Controller Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Running Simulation Steps
    print("\n[3/3] 运行前 5 步仿真...")
    idx = 0
    start_t = time.time()
    
    try:
        for data in loader.stream():
            step_start = time.time()
            result = controller.step(data)
            duration = time.time() - step_start
            
            print(f"\n  ⏱️ Step {idx+1} ({duration:.2f}s)")
            print(f"    Timestamp: {data.timestamp}")
            print(f"    Sensor DO: {data.base_do:.2f} mg/L, Temp: {data.water_temp:.1f} ℃")
            print(f"    Physics: DO Deficit={result['physics'].do_deficit:.2f}, R_fish={result['physics'].r_fish:.3f}")
            print(f"    Risk: {result['risk'].risk_level} (FishProb={result['risk'].fish_death_prob:.2f})")
            
            act = result['action']
            print(f"    Action: Aerator={'ON' if act.aerator_status else 'OFF'}, Light={'ON' if act.light_status else 'OFF'}")
            print(f"    Reason: {act.reason}")
            
            idx += 1
            if idx >= 5:
                break
                
        total_time = time.time() - start_t
        print(f"\n✅ 验证通过! 总耗时: {total_time:.2f}s (Avg: {total_time/5:.2f}s/step)")
        
    except Exception as e:
        print(f"\n❌ 仿真运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
