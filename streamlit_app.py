"""
BP DigitLab - Streamlit 主入口（训练页）。

这是一个非常薄的入口，只负责页面初始化和调用 app 层的编排模块。
所有复杂的业务逻辑都在 app/ 目录中。

使用方式：
    streamlit run streamlit_app.py
"""
from pathlib import Path

import streamlit as st

from app import bootstrap, state
from app.pages import train_page
from app.services.form_service import get_default_form_values

WORKSPACE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """主入口函数。"""
    # 页面初始化
    bootstrap.initialize_train_page()
    
    # 初始化 session state
    defaults = get_default_form_values(WORKSPACE_ROOT)
    state.ensure_form_initialized(defaults)
    state.ensure_ui_context_initialized(
        defaults["save_dir"],
        defaults["result_dir"]
    )
    
    # 渲染训练页
    train_page.render_train_page(WORKSPACE_ROOT)


if __name__ == "__main__":
    main()
