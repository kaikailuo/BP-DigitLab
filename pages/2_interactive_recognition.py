"""
BP DigitLab - Streamlit 多页面入口（识别页）。

这是 Streamlit 多页面模式中的第二个页面。
使用方式：
    streamlit run streamlit_app.py  # 然后在侧边栏选择这个页面
或者直接：
    streamlit run pages/2_interactive_recognition.py

所有复杂的业务逻辑都在 app/ 目录中。
"""
from pathlib import Path

import streamlit as st

from app import bootstrap, state
from app.pages import recognition_page

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """主入口函数。"""
    # 页面初始化
    bootstrap.initialize_recognition_page()
    
    # 初始化识别页特定的 session state
    state.ensure_recognition_state_initialized()
    
    # 渲染识别页
    recognition_page.render_recognition_page(WORKSPACE_ROOT)


if __name__ == "__main__":
    main()
