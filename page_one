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

def page_one():
    st.title("华南成本日报")

    # 添加日期筛选器
    start_date = st.sidebar.date_input("选择起始日期")
    end_date = st.sidebar.date_input("选择结束日期")

    files = st.sidebar.file_uploader("上传文件", type=['csv', 'xlsx'], accept_multiple_files=True)

    def process_file(file,start_date,end_date):
        """
        根据文件名处理文件
        :param file: 上传的文件对象
        """
        file_name = file.name
        results1_df = None
        results2_df = None
        results3_df = None
        results4_df = None
        
        if "图表" in file_name.lower():
            results2_df = process_daily_resource_file(file,start_date, end_date)
        elif "中转" in file_name.lower():
            results1_df = process_transfer_file(file,start_date, end_date)
        elif "收件" in file_name.lower():
            results3_df = process_pickup_file(file,start_date,end_date)
        elif "派件" in file_name.lower():
            results4_df = process_deliver_file(file,start_date,end_date)
        else:
            st.write(f"未知类型的文件: {file_name}")
            
        return results2_df,results1_df,results3_df,results4_df

    def process_transfer_file(file,start_date,end_date):
        df = pd.read_csv(file)
        
        # 将中转日期转换为日期时间类型
        df['中转日期'] = pd.to_datetime(df['中转日期'], format='%Y%m%d')

        df = df[(df['中转日期'] >= pd.to_datetime(start_date)) & (df['中转日期'] <= pd.to_datetime(end_date))]
        
        def calculate_figures(group_name, group_df):
            """
            计算指定组的中转票数、中转重量
            """
            
            tickets = group_df['中转票数'].sum()
            weight = group_df['中转重量'].sum()
            
            return tickets, weight
        
        results1 = []
        # 筛选大小件的数据
        big_df = df[df['是否大件场地'] == '大件']
        excluded_product_codes = ['SE000201', 'SE0114', 'SE0139', 'SE0155']
        big_df = big_df[~big_df['产品代码'].isin(excluded_product_codes)]
        
        all_big_tickets, all_big_weight = calculate_figures('全部', big_df)
        results1.append({
            '类型': '大件',
            '时间': '全部',
            '中转票数': all_big_tickets,
            '中转重量': all_big_weight
        })
        
        small_df = df[df['是否大件场地'] == '小件']
        
        all_small_tickets, all_small_weight = calculate_figures('全部', small_df)
        results1.append({
            '类型': '小件',
            '时间': '全部',
            '中转票数': all_small_tickets,
            '中转重量': all_small_weight
        })
        
        
        # 按周分组数据，每周开始时间为星期五
        weekly_big_grouped = big_df.groupby(pd.Grouper(key='中转日期', freq='W-FRI', label='left', closed='left'))

        # 遍历每周的数据并计算成本
        for week_start, week_df in weekly_big_grouped:
            week_start_str = week_start.strftime('%Y-%m-%d')
            week_end_str = (week_start + pd.DateOffset(days=6)).strftime('%Y-%m-%d')

            week_big_tickets,week_big_weight = calculate_figures('大件', week_df)
            
            results1.append({
                '类型': '大件',
                '时间': f"{week_start_str} 至 {week_end_str}",
                '中转票数': week_big_tickets,
                '中转重量': week_big_weight
            })

        # 按周分组数据，每周开始时间为星期五
        weekly_small_grouped = small_df.groupby(pd.Grouper(key='中转日期', freq='W-FRI', label='left', closed='left'))

        # 遍历每周的数据并计算成本
        for week_start, week_df in weekly_small_grouped:
            week_start_str = week_start.strftime('%Y-%m-%d')
            week_end_str = (week_start + pd.DateOffset(days=6)).strftime('%Y-%m-%d')

            week_small_tickets,week_small_weight = calculate_figures('小件', week_df)
            
            results1.append({
                '类型': '小件',
                '时间': f"{week_start_str} 至 {week_end_str}",
                '中转票数': week_small_tickets,
                '中转重量': week_small_weight
            })
        
        # 获取最晚5天的日期
        latest_date = df['中转日期'].max()
        latest_5_days = pd.date_range(start=latest_date - pd.Timedelta(days=4), end=latest_date)

        # 筛选出最晚5天大件的数据
        latest_5_days_big_data_df = df[df['中转日期'].isin(latest_5_days) & (df['是否大件场地'] == '大件') & ~df['产品代码'].isin(excluded_product_codes)]
        latest_5_days_big_grouped = latest_5_days_big_data_df.groupby(pd.Grouper(key='中转日期', freq='D'))

        # 按天输出指标
        for date, latest_5_days_df in latest_5_days_big_grouped:

            day_big_tickets, day_big_weight = calculate_figures('大件', latest_5_days_df)
            results1.append({
                '类型': '大件',
                '时间': date.strftime('%Y-%m-%d'),
                '中转票数': day_big_tickets,
                '中转重量': day_big_weight
            })
            
        # 筛选出最晚5天小件的数据
        latest_5_days_small_data_df = df[df['中转日期'].isin(latest_5_days) & (df['是否大件场地'] == '小件')]
        latest_5_days_small_grouped = latest_5_days_small_data_df.groupby(pd.Grouper(key='中转日期', freq='D'))
        
        # 按天输出指标
        for date, latest_5_days_df in latest_5_days_small_grouped:
            
            day_small_tickets, day_small_weight = calculate_figures('小件', latest_5_days_df)
            results1.append({
                '类型': '小件',
                '时间': date.strftime('%Y-%m-%d'),
                '中转票数': day_small_tickets,
                '中转重量': day_small_weight
            })
            
        
        # 将结果列表转换为 DataFrame
        results1_df = pd.DataFrame(results1)

        # 转置 DataFrame
        transposed_results1_df = results1_df.transpose()

        # 使用 st.dataframe 或 st.table 方法展示表格
        st.write("周度考核情况：")
        st.dataframe(transposed_results1_df)
        
        return results1_df




    def process_daily_resource_file(file,start_date,end_date):
        df = pd.read_csv(file)
            
        df = df[df['派驻组名称'] != '四川区派驻组']

        # 数据清洗：删除中转重量大于100的数据
        df = df[df['中转重量'] <= 100]

        # 将计划发车日期转换为日期时间类型
        df['计划发车日期'] = pd.to_datetime(df['计划发车日期'])

        df = df[(df['计划发车日期'] >= pd.to_datetime(start_date)) & (df['计划发车日期'] <= pd.to_datetime(end_date))]


        def calculate_costs(group_name, group_df, fixed_unique_days=None):
            """
            计算指定组的中转票数、中转重量、任务数、GRD账单总成本、总吨公里、线路里程
            """
            
            unique_days = group_df['计划发车日期'].dt.date.nunique()
            if fixed_unique_days is not None:
                unique_days = fixed_unique_days
            tickets = group_df['中转票数'].sum()
            weight = group_df['中转重量'].sum()
            tasks = group_df['任务数'].sum()
            total_grd_cost = group_df['GRD账单总成本'].sum()
            total_ton_km = group_df['总吨公里'].sum()
            total_line_km = group_df['线路里程'].sum()
            
            total_small_grd_cost = group_df.loc[(group_df['备份字段s11'] == '小件') & (group_df['运输等级（修正）'].isin(['一级运输', '二级运输'])), 'GRD账单总成本'].sum()
            total_grd_trunk_big_cost = group_df.loc[(group_df['备份字段s11'] == '大件') & (group_df['运输等级（修正）'].isin(['一级运输', '二级运输'])), 'GRD账单总成本'].sum()
            
            temporary_vehicle_average = group_df[group_df['交易渠道类型']=='临时交易']['任务数'].sum() / unique_days
            
            avg_tickets_per_day = tickets / unique_days
            avg_weight_per_day = weight / unique_days
            avg_tasks_per_day = tasks / unique_days
            ton_km_cost = total_grd_cost / total_ton_km
            kg_cost = total_grd_cost / (weight * 1000)
            km_cost = total_grd_cost / total_line_km
            
            return total_small_grd_cost, total_grd_trunk_big_cost, temporary_vehicle_average, avg_tickets_per_day, avg_weight_per_day, avg_tasks_per_day, ton_km_cost, kg_cost, km_cost
        
        # 创建一个 DataFrame 来存储每周的考核结果
        results2 = []
        
        # 筛选地区代码为111Y的数据
        region_df = df[df['地区代码'] == '111Y']
        all_small_grd_cost, all_grd_trunk_big_cost, all_temporary_vehicle_average, all_avg_tickets_per_day, all_avg_weight_per_day, all_avg_tasks_per_day, all_ton_km_cost, all_kg_cost, all_km_cost = calculate_costs('全部', region_df)
        
        results2.append({
            '时间': '全部',
            '平均每天中转票数': all_avg_tickets_per_day,
            '平均每天中转重量': all_avg_weight_per_day,
            '平均每天任务数': all_avg_tasks_per_day,
            '吨公里成本': all_ton_km_cost,
            '单公斤成本': all_kg_cost,
            '单公里成本': all_km_cost,
            '小件干线总成本': all_small_grd_cost,
            '大件干线总成本': all_grd_trunk_big_cost,
            '临时车日均交易次数': all_temporary_vehicle_average
        })

        # 按周分组数据，每周开始时间为星期五
        weekly_grouped = region_df.groupby(pd.Grouper(key='计划发车日期', freq='W-FRI', label='left', closed='left'))


        # 遍历每周的数据并计算成本
        for week_start, week_df in weekly_grouped:
            week_start_str = week_start.strftime('%Y-%m-%d')
            week_end_str = (week_start + pd.DateOffset(days=6)).strftime('%Y-%m-%d')

            week_small_grd_cost, week_grd_trunk_big_cost, week_temporary_vehicle_average, week_avg_tickets_per_day, week_avg_weight_per_day, week_avg_tasks_per_day, week_ton_km_cost, week_kg_cost, week_km_cost = calculate_costs('全部', week_df)

            # 将结果添加到列表中
            results2.append({
                '时间': f"{week_start_str} 至 {week_end_str}",
                '平均每天中转票数': week_avg_tickets_per_day,
                '平均每天中转重量': week_avg_weight_per_day,
                '平均每天任务数': week_avg_tasks_per_day,
                '吨公里成本': week_ton_km_cost,
                '单公斤成本': week_kg_cost,
                '单公里成本': week_km_cost,
                '小件干线总成本': week_small_grd_cost,
                '大件干线总成本': week_grd_trunk_big_cost,
                '临时车日均交易次数': week_temporary_vehicle_average
            })

        # 最近5日情况汇总
        # 获取最近5天的日期
        latest_date = df['计划发车日期'].max()
        date_range = pd.date_range(start=latest_date - pd.Timedelta(days=4), end=latest_date)

        # 按天分组数据并计算指标
        for date in date_range:
            # 选择当天的数据
            daily_df = df[df['计划发车日期'] == date]

            # 地区考核情况
            region_df = daily_df[daily_df['地区代码'] == '111Y']
            tickets = region_df['中转票数'].sum()
            weight = region_df['中转重量'].sum()
            tasks = region_df['任务数'].sum()
            
            total_grd_cost = region_df['GRD账单总成本'].sum()
            total_ton_km = region_df['总吨公里'].sum()
            total_line_km = region_df['线路里程'].sum()

            total_grd_small_cost = region_df[region_df['备份字段s11'] == '小件']['GRD账单总成本'].sum()
            total_grd_big_cost = region_df[(region_df['备份字段s11'] == '大件') & (region_df['运输等级（修正）'].isin(['一级运输', '二级运输']))]['GRD账单总成本'].sum()

            ton_km_cost = total_grd_cost / total_ton_km if total_ton_km > 0 else None
            kg_cost = total_grd_cost / (weight * 1000) if weight > 0 else None
            km_cost = total_grd_cost / total_line_km if total_line_km > 0 else None
            
            # 计算日均临时车交易次数
            daily_region_temporary_vehicle_df = region_df[region_df['交易渠道类型']=='临时交易']
            daily_temporary_vehicle_count = daily_region_temporary_vehicle_df['任务数'].sum()
            

            results2.append({
                '时间': date,
                '平均每天中转票数': tickets,
                '平均每天中转重量': weight,
                '平均每天任务数': tasks,
                '吨公里成本': ton_km_cost,
                '单公斤成本': kg_cost,
                '单公里成本': km_cost,
                '小件干线总成本': total_grd_small_cost,
                '大件干线总成本': total_grd_big_cost,
                '临时车日均交易次数': daily_temporary_vehicle_count,
            })

        # 创建DataFrame存储每日数据
        results2_df = pd.DataFrame(results2)


        # 转置 DataFrame
        transposed_results2_df = results2_df.transpose()

        # 使用 st.dataframe 或 st.table 方法展示表格
        st.write("周度考核情况：")
        st.dataframe(transposed_results2_df)

        return results2_df


    def process_pickup_file(file,start_date, end_date):
        # 读取Excel文件
        df = pd.read_excel(file)

        # 将日期列转换为 datetime 类型
        df['开始时间'] = pd.to_datetime(df['开始时间'])

        df = df[(df['开始时间'] >= pd.to_datetime(start_date)) & (df['开始时间'] <= pd.to_datetime(end_date))]

        results3 = []

        total_pickups = df['收件票数'].sum()
        total_days = len(df['开始时间'].dt.date.unique())

        total_avg_pickups = total_pickups / total_days
        results3.append({
            '时间': '全部',
            '收件票数': total_avg_pickups,
        })

        # 按周分组数据，每周开始时间为星期五
        weekly_grouped = df.groupby(pd.Grouper(key='开始时间', freq='W-FRI', label='left', closed='left'))

        # 遍历每周的开始日期，计算每周的收件票数和天数
        for week_start, group_data in weekly_grouped:
            week_start_str = week_start.strftime('%Y-%m-%d')
            week_end_str = (week_start + pd.DateOffset(days=6)).strftime('%Y-%m-%d')
            
            week_pickups = group_data['收件票数'].sum()
            week_days = len(group_data['开始时间'].dt.date.unique())
            
            if week_days == 0:
                week_avg_pickups = 0
            else:
                week_avg_pickups = week_pickups / week_days
            
            results3.append({
                '时间': f"{week_start_str} 至 {week_end_str}",
                '收件票数': week_avg_pickups,
            })
        
        latest_date = df['开始时间'].max()
        date_range = pd.date_range(start=latest_date - pd.Timedelta(days=4), end=latest_date)

        for date in date_range:
            # 选择当天的数据
            daily_df = df[df['开始时间'] == date]

            daily_pickups = daily_df['收件票数'].sum()
            results3.append({
                '时间': date,
                '收件票数': daily_pickups,
            })

        results3_df = pd.DataFrame(results3)

        transposed_results3_df = results3_df.transpose()

        st.dataframe(transposed_results3_df)
        return results3_df

    def process_deliver_file(file,start_date, end_date):
        # 读取Excel文件
        df = pd.read_excel(file)

        # 将日期列转换为 datetime 类型
        df['日期'] = pd.to_datetime(df['日期'])

        df = df[(df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))]

        results4 = []

        total_deliver = df['派件量'].sum()
        total_days = len(df['日期'].dt.date.unique())

        total_avg_deliver = total_deliver / total_days
        results4.append({
            '时间': '全部',
            '派件票数': total_avg_deliver,
        })

        # 按周分组数据，每周开始时间为星期五
        weekly_grouped = df.groupby(pd.Grouper(key='日期', freq='W-FRI', label='left', closed='left'))

        # 遍历每周的开始日期，计算每周的收件票数和天数
        for week_start, group_data in weekly_grouped:
            week_start_str = week_start.strftime('%Y-%m-%d')
            week_end_str = (week_start + pd.DateOffset(days=6)).strftime('%Y-%m-%d')
            
            week_deliver = group_data['派件量'].sum()
            week_days = len(group_data['日期'].dt.date.unique())
            
            if week_days == 0:
                week_avg_deliver = 0
            else:
                week_avg_deliver = week_deliver / week_days
            
            results4.append({
                '时间': f"{week_start_str} 至 {week_end_str}",
                '派件票数': week_avg_deliver,
            })
        

        latest_date = df['日期'].max()
        date_range = pd.date_range(start=latest_date - pd.Timedelta(days=4), end=latest_date)

        for date in date_range:
            # 选择当天的数据
            daily_df = df[df['日期'] == date]

            daily_deliver = daily_df['派件量'].sum()
            results4.append({
                '时间': date,
                '派件票数': daily_deliver,
            })

        results4_df = pd.DataFrame(results4)

        transposed_results4_df = results4_df.transpose()

        st.dataframe(transposed_results4_df)
        return results4_df


        
    # 处理每个上传的文件
    if files:
        # 初始化空列表来存储每个文件处理后的 DataFrame
        results1_dfs = []
        results2_dfs = []
        results3_dfs = []
        results4_dfs = []

        # 处理每个上传的文件
        for file in files:
            st.write(f"文件名: {file.name}")
            results2_df, results1_df,results3_df,results4_df = process_file(file,start_date,end_date)
            if results1_df is not None:
                results1_dfs.append(results1_df)
            if results2_df is not None:
                results2_dfs.append(results2_df)
            if results3_df is not None:
                results3_dfs.append(results3_df)
            if results4_df is not None:
                results4_dfs.append(results4_df)

        # 定义一个函数来处理时间列
        def process_time_column(time):
            try:
                # 尝试将时间转换为日期时间格式
                dt = pd.to_datetime(time)
                return dt.date()
            except ValueError:
                # 如果转换失败，返回原始值
                return time
        # 检查列表是否为空
        if results1_dfs and results2_dfs:
            # 合并数据
            results1_dfs = pd.concat(results1_dfs, ignore_index=True)
            results2_dfs = pd.concat(results2_dfs, ignore_index=True)
            # 处理时间列
            results1_dfs['时间'] = results1_dfs['时间'].apply(process_time_column)
            results2_dfs['时间'] = results2_dfs['时间'].apply(process_time_column)
            merged_df = pd.merge(results2_dfs,results1_dfs,  on='时间', how='outer')

            # 筛选大件数据并计算大件单公斤成本
            large_items = merged_df[merged_df['类型'] == '大件']
            large_items['大件单公斤成本'] = large_items['大件干线总成本'] / large_items['中转重量']

            # 筛选小件数据并计算小件单公斤成本
            small_items = merged_df[merged_df['类型'] == '小件']
            small_items['小件单票成本'] = small_items['小件干线总成本'] / small_items['中转票数']

            # 创建一个空的 DataFrame 来存储结果
            total_result_df = pd.DataFrame(columns=['时间', '大件单公斤成本', '小件单票成本'])

            # 将大件和小件的结果合并到 result_df 中
            for time in results1_dfs['时间']:
                large_cost = large_items[large_items['时间'] == time]['大件单公斤成本'].values[0] if not large_items[large_items['时间'] == time].empty else None
                small_cost = small_items[small_items['时间'] == time]['小件单票成本'].values[0] if not small_items[small_items['时间'] == time].empty else None
                total_result_df = pd.concat([total_result_df, pd.DataFrame({'时间': [time], '大件单公斤成本': [large_cost], '小件单票成本': [small_cost]})], ignore_index=True)

            # 合并结果到 df2
            results2_dfs = pd.merge(results2_dfs, total_result_df,on='时间', how='left')

            # 如果需要合并 results3_dfs，可以继续添加
            results3_dfs = pd.concat(results3_dfs, ignore_index=True)
            results3_dfs['时间'] = results3_dfs['时间'].apply(process_time_column)
            results2_dfs = pd.merge(results2_dfs, results3_dfs, on='时间', how='left')

            results4_dfs = pd.concat(results4_dfs, ignore_index=True)
            results4_dfs['时间'] = results4_dfs['时间'].apply(process_time_column)
            results2_dfs = pd.merge(results2_dfs, results4_dfs, on='时间', how='left')

            '''# 删除完全相同的重复列
            results2_dfs = results2_dfs.loc[:, ~results2_dfs.columns.duplicated()]'''

            results2_dfs = results2_dfs.drop_duplicates()

            # 转置 DataFrame
            transposed_results2_dfs = results2_dfs.transpose()

            # 使用 st.dataframe 或 st.table 方法展示表格
            st.write("周度考核情况：")
            st.dataframe(transposed_results2_dfs)# 打印结果
        else:
            st.write("没有有效的数据可以处理")
