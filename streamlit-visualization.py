import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import sys

from pages.utils.streamlit_utils import *

st.set_page_config(page_title="LLM Emission Tests", page_icon=':seedling:', layout="wide")

if __name__ == "__main__":
    init_session_states()
    sidebar()
    st.switch_page("pages/vllm_tests.py")