import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
from datetime import datetime
import sys
from pathlib import Path

# Add project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import integration modules
# Import integration modules
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))
try:
    from data_loader import DataLoader
    from controller import MPCController
    from interfaces import SensorData
except ImportError:
    st.error("Error importing modules. Please ensure all integration files are present.")
    st.stop()


st.set_page_config(
    page_title="Fish-Veg Symbiosis Smart Brain",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #09ab3b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_controller():
    return MPCController(site="红光")

@st.cache_resource
def get_data_loader():
    # Use relative path to data
    data_path = PROJECT_ROOT / "data" / "featured_红光.csv"
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.stop()
    return DataLoader(str(data_path), site_name="红光")

def main():
    st.title("🤖 鱼菜共生智慧工厂 — AI 决策大脑")
    st.caption("Powered by Chronos (时序预测) + PINN (物理反演) + Causal Inference (因果决策)")

    # Sidebar controls
    with st.sidebar:
        st.header("🎮 控制台")
        
        # State Management
        if 'running' not in st.session_state:
            st.session_state['running'] = False
            
        if st.button("▶️ 开始仿真"):
            st.session_state['running'] = True
            
        if st.button("⏹️ 停止"):
            st.session_state['running'] = False
            
        speed = st.slider("仿真速度 (delay sec)", 0.0, 2.0, 0.1)
        
        st.divider()
        st.subheader("⚙️ 系统状态")
        status_text = st.empty()
        if st.session_state['running']:
            status_text.success("System Running")
        else:
            status_text.info("System Ready")

    # Initialize Session State Variables
    if 'history' not in st.session_state:
        st.session_state['history'] = {
            'timestamp': [],
            'do': [], 'do_pred': [],
            'temp': [],
            'risk': [],
        }
    
    if 'data_generator' not in st.session_state:
        try:
            loader = get_data_loader()
            st.session_state['data_generator'] = loader.stream()
        except Exception as e:
            st.error(f"Failed to initialize Data Loader: {e}")
            st.stop()
        
    controller = get_controller()

    # Layout: Top Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_do = st.empty()
    with col2:
        metric_temp = st.empty()
    with col3:
        metric_risk = st.empty()

    # Layout: Charts
    col_main, col_detail = st.columns([2, 1])
    
    with col_main:
        st.subheader("📊 实时环境与预测 (Chronos)")
        chart_spot = st.empty()
        
    with col_detail:
        st.subheader("🛡️ 智能决策 (Causal + MPC)")
        action_spot = st.empty()
        st.subheader("🧬 物理参数 (PINN)")
        phys_spot = st.empty()

    # Simulation Logic
    if st.session_state['running']:
        try:
            # We fetch ONE step per app rerun? No, specific loop needed for continuous update
            # But continuous loop blocks UI. 
            # We use a placeholder and loop.
            
            # To allow "Stop" button to work, we verify session state in loop
            # But sidebar button press triggers re-run.
            # So a loop inside here will be interrupted by re-run when stop is pressed.
            
            # Use st.empty to update
            
            # Get next data point
            data = next(st.session_state['data_generator'])
            
            # Run Controller Step
            result = controller.step(data)
            
            # Update History
            hist = st.session_state['history']
            hist['timestamp'].append(data.timestamp)
            hist['do'].append(data.base_do)
            hist['temp'].append(data.water_temp)
            hist['risk'].append(1 if result['risk'].risk_level == "High" else 0)
            
            # Keep rolling window
            window_size = 50
            if len(hist['timestamp']) > window_size:
                for k in hist:
                    hist[k] = hist[k][-window_size:]
                    
            # Update Metrics UI
            metric_do.metric("当前溶氧 (DO)", f"{data.base_do:.2f} mg/L")
            metric_temp.metric("当前水温", f"{data.water_temp:.1f} ℃")
            
            risk_lvl = result['risk'].risk_level
            risk_color = "red" if risk_lvl=="High" else ("orange" if risk_lvl=="Medium" else "green")
            metric_risk.markdown(f"#### 风险等级: :{risk_color}[{risk_lvl}]")
            
            # Update Main Chart (Realtime + Pred)
            # Combine history and prediction for visualization
            df_hist = pd.DataFrame({
                'Time': hist['timestamp'],
                'DO': hist['do'],
                'Type': ['History'] * len(hist['timestamp'])
            })
            
            df_pred = pd.DataFrame()
            if result['forecast'] and len(result['forecast'].do_pred) > 0:
                df_pred = pd.DataFrame({
                    'Time': result['forecast'].target_dates,
                    'DO': result['forecast'].do_pred,
                    'Type': ['Prediction'] * len(result['forecast'].do_pred)
                })
            
            df_chart = pd.concat([df_hist, df_pred])
            
            c = alt.Chart(df_chart).mark_line(point=True).encode(
                x='Time',
                y=alt.Y('DO', scale=alt.Scale(domain=[0, 10])),
                color='Type',
                tooltip=['Time', 'DO', 'Type']
            ).properties(height=350).interactive()
            
            chart_spot.altair_chart(c, use_container_width=True)
            
            # Update Detail Panels
            act = result['action']
            status_icon = "🟢 RUNNING" if act.aerator_status else "⚪ OFF"
            action_spot.success(f"""
            **增氧机**: {status_icon} ({act.aerator_duration}h)
            
            **补光灯**: {"💡 ON" if act.light_status else "⚫ OFF"}
            
            **原因**: {act.reason}
            """)
            
            phys = result['physics']
            phys_spot.info(f"""
            - K_La: {phys.kla:.3f}
            - R_fish: {phys.r_fish:.3f}
            - DO 亏损: {phys.do_deficit:.2f}
            """)
            
            time.sleep(speed)
            st.rerun() # Rerun to loop
            
        except StopIteration:
            st.success("Simulation Completed (End of Data)")
            st.session_state['running'] = False
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state['running'] = False

if __name__ == "__main__":
    main()
