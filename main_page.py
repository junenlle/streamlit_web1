# -*- coding: utf-8 -*-
"""
运营规划报表
"""

import os
import streamlit as st
import pandas as pd

from page_one import page_one
from page_two import page_two
from page_three import page_three# -*- coding: utf-8 -*-
"""
运营规划报表
"""

import os
import streamlit as st
import pandas as pd

from page_one import page_one
from page_two import page_two
from page_three import page_three
from page_five import page_five
from page_eight import page_eight
from page_ten import page_ten
from page_five_pytorch import page_five_pytorch

# 获取当前页面
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["main"])[0]

# 左侧边栏导航
with st.sidebar:  
    if st.button("华南成本日报"):
        st.experimental_set_query_params(page="one")
    
    if st.button("车辆放空监控"):
        st.experimental_set_query_params(page="two")
    
    if st.button("干线装载率早晚报"):
        st.experimental_set_query_params(page="three")
    
    if st.button("收派件量预测"):
        st.experimental_set_query_params(page="five")
    
    if st.button("成本日报-周一版"):
        st.experimental_set_query_params(page="eight")
    
    if st.button("收派件量预测-XGBoost"):
        st.experimental_set_query_params(page="ten")
    
    if st.button("收派件量预测-Pytorch"):
        st.experimental_set_query_params(page="five_pytorch")



# 页面选择器
if page == "main":
    st.title("欢迎来到运营规划报表主页")
elif page == "one":
    page_one()
elif page == "two":
    page_two()
elif page == "three":
    page_three()
elif page == "five":
    page_five()
elif page == "eight":
    page_eight()
elif page == "ten":
    page_ten()
elif page == "five_pytorch":
    page_five_pytorch()







# 获取当前页面
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["main"])[0]

# 左侧边栏导航
with st.sidebar:  
    if st.button("华南成本日报"):
        st.experimental_set_query_params(page="one")
    
    if st.button("车辆放空监控"):
        st.experimental_set_query_params(page="two")
    
    if st.button("干线装载率早晚报"):
        st.experimental_set_query_params(page="three")

# 页面选择器
if page == "main":
    st.title("欢迎来到运营规划报表主页")
elif page == "one":
    page_one()
elif page == "two":
    page_two()
elif page == "three":
    page_three()
