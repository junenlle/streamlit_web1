# -*- coding: utf-8 -*-
"""
运营规划报表
"""

import os
import streamlit as st
import pandas as pd

from page_one import page_one
from page_eight import page_eight

# 获取当前页面
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["main"])[0]

# 左侧边栏导航
with st.sidebar:  
    if st.button("华南成本日报"):
        st.experimental_set_query_params(page="one")
    
    if st.button("成本日报-周一版"):
        st.experimental_set_query_params(page="eight")



# 页面选择器
if page == "main":
    st.title("欢迎来到运营规划报表主页")
elif page == "one":
    page_one()
elif page == "eight":
    page_eight()
