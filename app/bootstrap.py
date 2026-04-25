"""
Streamlit 应用启动与配置。

负责页面配置、全局样式、依赖初始化等启动级别的工作。
"""
import streamlit as st


def setup_page_config(page_title: str = "BP DigitLab", layout: str = "wide") -> None:
    """
    配置 Streamlit 页面。
    
    参数：
        page_title: 页面标题
        layout: 布局模式 (wide/centered)
    """
    st.set_page_config(page_title=page_title, layout=layout)


def setup_global_styles() -> None:
    """注入全局 CSS 样式。"""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"] {
            font-size: 1.18rem !important;
            font-weight: 700;
            min-height: 2.7rem;
            padding: 0.8rem 0.9rem !important;
            border-radius: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_recognition_page_styles() -> None:
    """注入识别页专用的 CSS 样式。"""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [data-testid="stSidebarNavLink"] {
            font-size: 1.18rem !important;
            font-weight: 700;
            min-height: 2.7rem;
            padding: 0.8rem 0.9rem !important;
            border-radius: 0.6rem;
        }
        div[data-testid="stButton"] button#back-to-train {
            font-size: 1.05rem;
            font-weight: 600;
            padding-top: 0.65rem;
            padding-bottom: 0.65rem;
        }
        .predict-digit {
            font-size: 4rem;
            font-weight: 800;
            line-height: 1;
            margin: 0.25rem 0 0.75rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_train_page() -> None:
    """初始化训练页。"""
    setup_page_config(page_title="BP DigitLab - 训练模型", layout="wide")
    setup_global_styles()


def initialize_recognition_page() -> None:
    """初始化识别页。"""
    setup_page_config(page_title="BP DigitLab - 交互识别", layout="wide")
    setup_recognition_page_styles()
