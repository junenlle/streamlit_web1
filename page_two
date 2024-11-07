import os
import streamlit as st
import pandas as pd

def get_file_list(suffix, path):
    """
    获取当前目录所有指定后缀的文件名列表和绝对路径列表
    :param suffix: 文件后缀
    :param path: 搜索路径
    :return: 文件名列表, 文件路径列表
    """
    file_names = []
    file_paths = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if os.path.splitext(name)[1] == suffix:
                file_names.append(name)
                file_paths.append(os.path.join(root, name))
    return file_names, file_paths

def page_two():
    st.title("车辆放空监控")

    file = st.sidebar.file_uploader("上传文件", type=['csv', 'xlsx'])

    if file is not None:
        # 读取 CSV 文件
        df = pd.read_csv(file)

        # 将日期列转换为日期格式
        df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')
        df.sort_values(by='日期', inplace=True)

        # 定义片区划分规则
        def assign_region(code):
            if code in ['769WB', '769WF', '769WK', '769WM', '769WS', '769XG', '769WV', '769VZ', '769DCR']:
                return '东莞片'
            elif code in ['757WA', '757WH', '757WL', '757WE', '758CC001', '763X', '763XD']:
                return '佛山片'
            elif code in ['020RD', '020W', '020WG', '020WM', '020WS', '020XB', '020CC006', '020CC001', '020R', '020CC008', '020XK', '020WT', '020Z', '020WE', '020VM', '020WK', '020RH', '751VA']:
                return '广州片'
            elif code in ['755VF', '755W', '755WE', '755WF', '755WM', '755X', '755R', '755RC', '755VM', '755RH', '755Z', '755WJ', '755ZS']:
                return '深圳片'
            elif code in ['663WA', '663WH', '752WA', '752WH', '762VA', '762VH', '663R', '660VH', '768WA', '663WB', '768VA', '753WH', '660VW']:
                return '粤东片'
            elif code in ['750W', '756W', '760W', '750VA', '760WH', '756R', '759VA', '668VH', '750WA', '668VA', '759RHZ13586', '759RHZ13578', '759VAHZ15425']:
                return '粤西片'
            else:
                return '未知'

        # 应用片区划分规则
        df['片区'] = df['任务网点'].apply(assign_region)

        # 定义固定的片区顺序列表
        fixed_regions = ['东莞片', '佛山片', '广州片', '深圳片', '粤东片', '粤西片']

        # 动态日期选择函数
        def get_region_stats(date=None, date_range=None):
            """
            根据提供的日期或日期范围返回每个片区的统计数据。
            
            :param date: 单个日期，格式为 'YYYY-MM-DD'
            :param date_range: 日期范围，格式为 ('start_date', 'end_date')
            :return: 包含统计数据的DataFrame
            """
            if date is not None:
                date = pd.to_datetime(date)
                filtered_df = df[df['日期'] == date]
            elif date_range is not None:
                start_date, end_date = [pd.to_datetime(d) for d in date_range]
                filtered_df = df[(df['日期'] >= start_date) & (df['日期'] <= end_date)]
            else:
                raise ValueError("Either 'date' or 'date_range' must be provided.")
            
            # 统计数据
            stats_by_region = filtered_df.groupby('片区').agg({'计划需求ID': 'nunique', '应补偿': 'sum'}).rename(columns={'计划需求ID': '放空车辆数量', '应补偿': '应补偿金额'})
            
            # 重新索引以确保固定顺序，并填充缺失值
            stats_by_region = stats_by_region.reindex(fixed_regions).fillna(0)
            
            # 按始发网点统计
            stats_by_origin = filtered_df.groupby('任务网点').agg({'计划需求ID': 'nunique', '应补偿': 'sum'}).rename(columns={'计划需求ID': '放空车辆数量', '应补偿': '应补偿金额'})
            
            return stats_by_region, stats_by_origin

        # 获取数据表中最早的日期和尾部日期
        earliest_date = df['日期'].min().date()
        latest_date = df['日期'].max().date()

        # 日期筛选器
        date_option = st.sidebar.radio("选择日期筛选方式", ("单个日期", "日期范围"))

        if date_option == "单个日期":
            selected_date = st.sidebar.date_input("选择日期", min_value=earliest_date, max_value=latest_date, value=earliest_date)
            stats_by_region, stats_by_origin = get_region_stats(date=selected_date)
        else:
            selected_date_range = st.sidebar.slider("选择日期范围", min_value=earliest_date, max_value=latest_date, value=(earliest_date, latest_date))
            stats_by_region, stats_by_origin = get_region_stats(date_range=selected_date_range)

        # 显示结果
        st.write(f"日期筛选：{date_option}")
        st.write(f"按片区统计：")
        st.write(stats_by_region)
        st.write("-" * 40)
        st.write(f"按任务网点统计：")
        st.write(stats_by_origin)
        st.write("-" * 40)

    else:
        st.write("请上传文件以开始分析。")
