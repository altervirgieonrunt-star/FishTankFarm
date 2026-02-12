import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "code" / "5.系统集成"))

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QFrame, 
                             QGridLayout, QSizePolicy, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QFont, QColor

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

# Import Logic Modules
try:
    from data_loader import DataLoader
    from controller import MPCController
    from gui.styles import Styles
except ImportError as e:
    print(f"Import Error: {e}")
    # Handle imports when running from different contexts if needed
    pass

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Apply style before creating figure
        for k, v in Styles.get_mpl_style().items():
            matplotlib.rcParams[k] = v
            
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # Remove top and right spines for a cleaner look
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        
        super(MplCanvas, self).__init__(self.fig)
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鱼菜共生智慧工厂 — AI 决策大脑")
        self.resize(1400, 950)
        
        # Apply Global QSS
        self.setStyleSheet(Styles.STYLESHEET)
        
        # --- Logic & Data ---
        try:
            data_path = PROJECT_ROOT / "data" / "featured_红光.csv"
            self.loader = DataLoader(str(data_path), site_name="红光")
            self.data_generator = self.loader.stream()
            self.controller = MPCController(site="红光")
        except Exception as e:
            print(f"Initialization Error: {e}")
            # For UI testing without data, we might want a fallback, but here we exit
            sys.exit(1)
            
        self.history = {
            'timestamp': [],
            'do': [],
            'temp': [],
            'risk': []
        }
        
        # --- UI Initialization ---
        self.init_ui()
        
        # --- Timer for Simulation ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_simulation_step)
        self.simulation_speed = 500 # Default ms (0.5s)

    def add_shadow(self, widget, blur=15, alpha=30, offset=4):
        """Helper to add soft shadow to widgets"""
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur)
        shadow.setColor(QColor(0, 0, 0, alpha))
        shadow.setOffset(0, offset)
        widget.setGraphicsEffect(shadow)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout (HBox: Sidebar | Content)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ================= Left Sidebar (Controls) =================
        sidebar = QFrame()
        sidebar.setObjectName("sidebar") # For QSS
        sidebar.setFixedWidth(300)
        
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(30, 40, 30, 30)
        sidebar_layout.setSpacing(20)
        
        # Branding / Title
        lbl_title = QLabel("🌿 鱼菜共生\nAI 决策系统")
        lbl_title.setObjectName("sidebar_title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sidebar_layout.addWidget(lbl_title)
        
        sidebar_layout.addSpacing(20)
        
        # Controls Section
        lbl_control = QLabel("仿真控制")
        lbl_control.setObjectName("sidebar_label")
        sidebar_layout.addWidget(lbl_control)
        
        self.btn_start = QPushButton("开始仿真")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_simulation)
        sidebar_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("停止仿真")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_stop.setEnabled(False) 
        sidebar_layout.addWidget(self.btn_stop)
        
        sidebar_layout.addSpacing(20)
        
        # Speed Control
        lbl_speed = QLabel("仿真速度")
        lbl_speed.setObjectName("sidebar_label")
        sidebar_layout.addWidget(lbl_speed)
        
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setMinimum(100)
        self.slider_speed.setMaximum(2000)
        self.slider_speed.setValue(500)
        self.slider_speed.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider_speed.valueChanged.connect(self.update_speed)
        sidebar_layout.addWidget(self.slider_speed)
        
        sidebar_layout.addStretch()
        
        # Status Footer
        self.lbl_status = QLabel("● 系统就绪")
        self.lbl_status.setObjectName("status_bar")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft)
        sidebar_layout.addWidget(self.lbl_status)
        
        main_layout.addWidget(sidebar)
        
        # ================= Center Panel (Dashboard) =================
        content_panel = QWidget()
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(25)
        
        # --- Top Metrics Row ---
        metrics_container = QWidget()
        metrics_layout = QHBoxLayout(metrics_container)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(25)
        
        # Metric 1: DO
        self.metric_do = self.create_metric_card("溶解氧 (DO)", "0.00", "mg/L")
        metrics_layout.addWidget(self.metric_do)
        
        # Metric 2: Temp
        self.metric_temp = self.create_metric_card("水体温度", "0.0", "℃")
        metrics_layout.addWidget(self.metric_temp)
        
        # Metric 3: Risk
        self.metric_risk = self.create_metric_card("当前风险等级", "Low", "", is_risk=True)
        metrics_layout.addWidget(self.metric_risk)
        
        content_layout.addWidget(metrics_container)
        
        # --- Middle Section: Chart + Details ---
        mid_section = QWidget()
        mid_layout = QHBoxLayout(mid_section)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(25)
        
        # 1. Main Chart (Takes up 66%)
        chart_frame = QFrame()
        chart_frame.setProperty("class", "card") # QSS class
        self.add_shadow(chart_frame)
        
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_chart_title = QLabel("实时环境监测与 AI 预测")
        lbl_chart_title.setProperty("class", "card_title")
        chart_layout.addWidget(lbl_chart_title)
        
        self.chart = MplCanvas(self, width=8, height=5, dpi=100)
        chart_layout.addWidget(self.chart)
        
        mid_layout.addWidget(chart_frame, stretch=2)
        
        # 2. Details Panel (Takes up 33%)
        details_col = QWidget()
        details_col_layout = QVBoxLayout(details_col)
        details_col_layout.setContentsMargins(0, 0, 0, 0)
        details_col_layout.setSpacing(25)
        
        # Decision Card
        decision_frame = QFrame()
        decision_frame.setProperty("class", "card")
        self.add_shadow(decision_frame)
        decision_layout = QVBoxLayout(decision_frame)
        decision_layout.setContentsMargins(20, 20, 20, 20)
        decision_layout.setSpacing(10)
        
        lbl_dec = QLabel("智能控制决策")
        lbl_dec.setProperty("class", "detail_header")
        decision_layout.addWidget(lbl_dec)
        
        self.lbl_aerator = QLabel("⚙️ 增氧机: 待机")
        self.lbl_aerator.setProperty("class", "detail_item")
        
        self.lbl_light = QLabel("💡 补光灯: 关闭")
        self.lbl_light.setProperty("class", "detail_item")
        
        self.lbl_reason = QLabel("📝 决策依据: 系统初始化完成")
        self.lbl_reason.setProperty("class", "detail_item")
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setStyleSheet("color: #6b7280; font-size: 13px; margin-top: 5px;")
        
        decision_layout.addWidget(self.lbl_aerator)
        decision_layout.addWidget(self.lbl_light)
        decision_layout.addWidget(self.lbl_reason)
        decision_layout.addStretch()
        
        details_col_layout.addWidget(decision_frame)
        
        # Physics Card
        phys_frame = QFrame()
        phys_frame.setProperty("class", "card")
        self.add_shadow(phys_frame)
        phys_layout = QVBoxLayout(phys_frame)
        phys_layout.setContentsMargins(20, 20, 20, 20)
        phys_layout.setSpacing(10)
        
        lbl_phys = QLabel("PINN 物理参数反演")
        lbl_phys.setProperty("class", "detail_header")
        phys_layout.addWidget(lbl_phys)
        
        self.lbl_kla = QLabel("K_La (氧传质系数): -")
        self.lbl_kla.setProperty("class", "detail_item")
        
        self.lbl_rfish = QLabel("R_fish (鱼耗氧率): -")
        self.lbl_rfish.setProperty("class", "detail_item")
        
        self.lbl_deficit = QLabel("DO Deficit (亏损): -")
        self.lbl_deficit.setProperty("class", "detail_item")
        
        phys_layout.addWidget(self.lbl_kla)
        phys_layout.addWidget(self.lbl_rfish)
        phys_layout.addWidget(self.lbl_deficit)
        phys_layout.addStretch()
        
        details_col_layout.addWidget(phys_frame)
        
        mid_layout.addWidget(details_col, stretch=1)
        
        content_layout.addWidget(mid_section)
        
        # Add content panel to main
        main_layout.addWidget(content_panel)

    def create_metric_card(self, title, initial_value, unit="", is_risk=False):
        frame = QFrame()
        frame.setProperty("class", "card") # Apply QSS class
        self.add_shadow(frame, blur=10, alpha=20)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_title = QLabel(title)
        lbl_title.setProperty("class", "card_title")
        
        val_container = QWidget()
        val_layout = QHBoxLayout(val_container)
        val_layout.setContentsMargins(0, 0, 0, 0)
        val_layout.setSpacing(5)
        
        lbl_val = QLabel(initial_value)
        lbl_val.setObjectName("value_label") # Store for update
        lbl_val.setProperty("class", "card_value")
        
        if is_risk:
            lbl_val.setProperty("risk", "Low")
        
        val_layout.addWidget(lbl_val)
        
        if unit:
            lbl_unit = QLabel(unit)
            lbl_unit.setStyleSheet(f"color: #9ca3af; font-size: 14px; font-weight: bold; margin-bottom: 2px;")
            lbl_unit.setAlignment(Qt.AlignmentFlag.AlignBottom)
            val_layout.addWidget(lbl_unit)
            
        val_layout.addStretch()
        
        layout.addWidget(lbl_title)
        layout.addWidget(val_container)
        return frame

    def update_metric(self, card_widget, text, risk_level=None):
        label = card_widget.findChild(QLabel, "value_label")
        if label:
            label.setText(text)
            if risk_level:
                label.setProperty("risk", risk_level)
                # Force style re-polish for dynamic property change
                label.style().unpolish(label)
                label.style().polish(label)

    def start_simulation(self):
        self.timer.start(self.simulation_speed)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("● 仿真运行中...")
        self.lbl_status.setStyleSheet("color: #10b981;") # Green

    def stop_simulation(self):
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("● 仿真已暂停")
        self.lbl_status.setStyleSheet("color: #f59e0b;") # Orange

    def update_speed(self, val):
        self.simulation_speed = val
        if self.timer.isActive():
            self.timer.setInterval(self.simulation_speed)

    def run_simulation_step(self):
        try:
            data = next(self.data_generator)
            
            # DEBUG: Track DO values
            print(f"[DEBUG] Time: {data.timestamp}, DO: {data.base_do:.2f}, Temp: {data.water_temp:.1f}")
            
            result = self.controller.step(data)
            
            # Update History
            self.history['timestamp'].append(pd.to_datetime(data.timestamp))
            self.history['do'].append(data.base_do)
            
            window = 50
            if len(self.history['timestamp']) > window:
                self.history['timestamp'] = self.history['timestamp'][-window:]
                self.history['do'] = self.history['do'][-window:]
            
            # Update Metrics
            self.update_metric(self.metric_do, f"{data.base_do:.2f}")
            self.update_metric(self.metric_temp, f"{data.water_temp:.1f}")
            
            risk_lvl = result['risk'].risk_level
            self.update_metric(self.metric_risk, risk_lvl, risk_level=risk_lvl)
            
            # Update Charts
            self.update_chart(result['forecast'])
            
            # Update Decisions
            act = result['action']
            ae_st = "🟢 开启" if act.aerator_status else "⚪ 待机"
            light_st = "💡 开启" if act.light_status else "⚫ 关闭"
            self.lbl_aerator.setText(f"⚙️ 增氧机: {ae_st} (时长 {act.aerator_duration:.1f}h)")
            self.lbl_light.setText(f"💡 补光灯: {light_st}")
            self.lbl_reason.setText(f"📝 依据: {act.reason}")
            
            # Update Physics
            phys = result['physics']
            self.lbl_kla.setText(f"K_La: {phys.kla:.3f}")
            self.lbl_rfish.setText(f"R_fish: {phys.r_fish:.3f}")
            self.lbl_deficit.setText(f"DO 亏损: {phys.do_deficit:.2f}")
            
        except StopIteration:
            self.stop_simulation()
            self.lbl_status.setText("● 数据播放结束")
        except Exception as e:
            print(f"Simulation Error: {e}")
            self.stop_simulation()
            self.lbl_status.setText(f"● 错误: {str(e)}")

    def update_chart(self, forecast):
        self.chart.axes.cla()
        
        # Apply grid again after cla()
        self.chart.axes.grid(True, linestyle='--', alpha=0.4, color='#e5e7eb')
        
        # Plot History
        times = self.history['timestamp']
        dos = self.history['do']
        
        # Use theme colors
        self.chart.axes.plot(times, dos, 'o-', label='History', color=Styles.COLOR_SECONDARY, linewidth=2, markersize=4, alpha=0.8)
        
        # Plot Prediction
        if forecast and len(forecast.do_pred) > 0:
            pred_times = pd.to_datetime(forecast.target_dates)
            pred_dos = np.array(forecast.do_pred) # Ensure it is a numpy array
            self.chart.axes.plot(pred_times, pred_dos, '--', label='AI Forecast', color=Styles.COLOR_PRIMARY, linewidth=2, markersize=0)
            
            # Fill area for prediction confidence (Simulated visual effect)
            self.chart.axes.fill_between(pred_times, pred_dos - 0.5, pred_dos + 0.5, color=Styles.COLOR_PRIMARY, alpha=0.1)

            # Connect line
            if len(times) > 0:
                self.chart.axes.plot([times[-1], pred_times[0]], [dos[-1], pred_dos[0]], '--', color=Styles.COLOR_PRIMARY, alpha=0.5)

        self.chart.axes.set_ylabel("Dissolved Oxygen (mg/L)")
        self.chart.axes.legend(loc='upper right', frameon=False)
        self.chart.fig.autofmt_xdate()
        self.chart.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Enable High DPI scaling
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())