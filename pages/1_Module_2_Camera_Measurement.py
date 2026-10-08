import os
import runpy
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "cv-module2-assignment"


def run_module_app() -> None:
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    original_set_page_config = st.set_page_config

    try:
        os.chdir(MODULE_ROOT)
        sys.path.insert(0, str(MODULE_ROOT))
        st.set_page_config = lambda *args, **kwargs: None
        runpy.run_path(str(MODULE_ROOT / "app.py"), run_name="__main__")
    finally:
        st.set_page_config = original_set_page_config
        sys.path[:] = original_path
        os.chdir(original_cwd)


run_module_app()

