class Styles:
    # --- Palette (Eco-Tech) ---
    COLOR_BG = "#f0f2f5"       # Main Window Background
    COLOR_SIDEBAR = "#ffffff"  # Sidebar Background
    COLOR_CARD = "#ffffff"     # Card Background
    
    COLOR_PRIMARY = "#059669"  # Emerald 600 (Primary Green)
    COLOR_SECONDARY = "#3b82f6" # Blue 500 (Secondary Blue)
    COLOR_ACCENT = "#8b5cf6"   # Violet (Accent)
    
    COLOR_TEXT_MAIN = "#1f2937" # Gray 900
    COLOR_TEXT_SUB = "#6b7280"  # Gray 500
    
    COLOR_SUCCESS = "#10b981"
    COLOR_WARNING = "#f59e0b"
    COLOR_DANGER = "#ef4444"
    COLOR_INFO = "#3b82f6"

    # --- Fonts ---
    FONT_FAMILY = "Segoe UI, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    FONT_HEADER_SIZE = "18px"
    FONT_SUBHEADER_SIZE = "14px"
    FONT_VALUE_SIZE = "28px"
    
    # --- QSS Stylesheet ---
    # This stylesheet covers the entire application
    STYLESHEET = f"""
    QMainWindow {{
        background-color: {COLOR_BG};
    }}

    /* --- Sidebar --- */
    QFrame#sidebar {{
        background-color: {COLOR_SIDEBAR};
        border-right: 1px solid #e5e7eb;
        border-radius: 0px; 
    }}
    
    QLabel#sidebar_title {{
        font-family: "{FONT_FAMILY}";
        font-size: 20px;
        font-weight: bold;
        color: {COLOR_PRIMARY};
        padding: 10px 0;
    }}
    
    QLabel#sidebar_label {{
        font-family: "{FONT_FAMILY}";
        font-size: 14px;
        font-weight: 600;
        color: {COLOR_TEXT_MAIN};
        margin-top: 10px;
    }}

    /* --- Buttons --- */
    QPushButton {{
        border-radius: 8px;
        padding: 12px 20px;
        font-family: "{FONT_FAMILY}";
        font-size: 14px;
        font-weight: bold;
        border: none;
    }}
    
    QPushButton#btn_start {{
        background-color: {COLOR_SUCCESS};
        color: white;
    }}
    QPushButton#btn_start:hover {{
        background-color: #059669;
    }}
    QPushButton#btn_start:pressed {{
        background-color: #047857;
    }}
    QPushButton#btn_start:disabled {{
        background-color: #d1fae5;
        color: #065f46;
    }}

    QPushButton#btn_stop {{
        background-color: {COLOR_BG}; /* Ghost button style for secondary */
        border: 2px solid {COLOR_DANGER};
        color: {COLOR_DANGER};
    }}
    QPushButton#btn_stop:hover {{
        background-color: #fef2f2;
    }}
    QPushButton#btn_stop:pressed {{
        background-color: #fee2e2;
    }}
    QPushButton#btn_stop:disabled {{
        border-color: #e5e7eb;
        color: #9ca3af;
        background-color: transparent;
    }}

    /* --- Sliders --- */
    QSlider::groove:horizontal {{
        border: 1px solid #e5e7eb;
        height: 6px;
        background: #f3f4f6;
        margin: 2px 0;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {COLOR_PRIMARY};
        border: 1px solid {COLOR_PRIMARY};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    
    /* --- Cards --- */
    QFrame.card {{
        background-color: {COLOR_CARD};
        border-radius: 16px;
        border: 1px solid #f3f4f6;
    }}
    
    /* --- Labels in Cards --- */
    QLabel.card_title {{
        font-family: "{FONT_FAMILY}";
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: {COLOR_TEXT_SUB};
    }}
    
    QLabel.card_value {{
        font-family: "{FONT_FAMILY}";
        font-size: {FONT_VALUE_SIZE};
        font-weight: 800;
        color: {COLOR_TEXT_MAIN};
    }}
    
    /* Risk Levels */
    QLabel[risk="Low"] {{ color: {COLOR_SUCCESS}; }}
    QLabel[risk="Medium"] {{ color: {COLOR_WARNING}; }}
    QLabel[risk="High"] {{ color: {COLOR_DANGER}; }}
    
    /* --- Right Panel Details --- */
    QLabel.detail_header {{
        font-family: "{FONT_FAMILY}";
        font-size: 16px;
        font-weight: 700;
        color: {COLOR_TEXT_MAIN};
        padding-bottom: 5px;
        border-bottom: 2px solid {COLOR_BG};
    }}
    
    QLabel.detail_item {{
        font-family: "{FONT_FAMILY}";
        font-size: 14px;
        color: {COLOR_TEXT_MAIN};
        padding: 4px 0;
    }}

    /* --- Footer Status --- */
    QLabel#status_bar {{
        font-family: "{FONT_FAMILY}";
        font-size: 12px;
        color: {COLOR_TEXT_SUB};
        padding: 5px;
    }}
    """
    
    @staticmethod
    def get_mpl_style():
        """Returns a dictionary for matplotlib rcParams to match the theme."""
        return {
            'figure.facecolor': Styles.COLOR_CARD,
            'axes.facecolor': Styles.COLOR_CARD,
            'axes.edgecolor': '#e5e7eb',
            'axes.labelcolor': Styles.COLOR_TEXT_SUB,
            'text.color': Styles.COLOR_TEXT_MAIN,
            'xtick.color': Styles.COLOR_TEXT_SUB,
            'ytick.color': Styles.COLOR_TEXT_SUB,
            'grid.color': '#f3f4f6',
            'font.family': 'sans-serif',
            'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial'],
            'font.size': 10,
            'lines.linewidth': 2.5
        }