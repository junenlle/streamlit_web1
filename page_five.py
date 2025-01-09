import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 设置字体为楷体
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 楷体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号问题

def page_five():

    # 添加文件上传组件
    uploaded_file = st.file_uploader("上传 Excel 文件", type=["xlsx"])

    if uploaded_file is not None:
        # 读取 Excel 文件
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # 确保列名与代码中使用的列名一致
        df.columns = ['date', 'pickups', 'delivers']

        # 将Excel日期序列号转换为标准日期格式
        df['date'] = pd.to_datetime(df['date'], unit='D', origin='1899-12-30')

        # 设置日期列为索引
        df.set_index('date', inplace=True)

        # 定义多个大高峰日期范围
        major_peak_ranges = [
            pd.date_range(start='2024-05-21', end='2024-05-22'),
            pd.date_range(start='2024-06-18', end='2024-06-19'), #618
            pd.date_range(start='2024-10-22', end='2024-10-25'),
            pd.date_range(start='2024-11-01', end='2024-11-01'),
            pd.date_range(start='2024-11-04', end='2024-11-04'),
            pd.date_range(start='2024-11-11', end='2024-11-12'), #双十一
            pd.date_range(start='2024-12-09', end='2024-12-13'), #双12
        ]

        # 合并所有日期范围
        major_peak_dates = pd.DatetimeIndex([date for range_ in major_peak_ranges for date in range_])

        # 小高峰日期范围
        minor_peak_ranges = [
            pd.date_range(start='2024-01-29', end='2024-02-01'),
            pd.date_range(start='2024-09-09', end='2024-09-13'), #双9
            pd.date_range(start='2024-12-13', end='2024-12-14'),
        ]

        minor_peak_dates = pd.DatetimeIndex([date for range_ in minor_peak_ranges for date in range_])

        # 节假日日期范围
        holiday_ranges =  [
            pd.date_range(start='2024-01-01', end='2024-01-01'), #元旦
            pd.date_range(start='2024-02-07', end='2024-02-17'), #春节
            pd.date_range(start='2024-04-04', end='2024-04-05'), #清明
            pd.date_range(start='2024-05-01', end='2024-05-03'), #劳动节
            pd.date_range(start='2024-06-09', end='2024-06-10'), #端午
            pd.date_range(start='2024-09-15', end='2024-09-17'), #中秋
            pd.date_range(start='2024-10-01', end='2024-10-07'), #国庆
        ]

        holiday_dates = pd.DatetimeIndex([date for range_ in holiday_ranges for date in range_])

        # 添加大高峰、小高峰、节假日和工作日特征
        df['period'] = 'normal'
        df.loc[df.index.isin(major_peak_dates), 'period'] = 'major_peak'
        df.loc[df.index.isin(minor_peak_dates), 'period'] = 'minor_peak'
        df.loc[df.index.isin(holiday_dates), 'period'] = 'holiday'
        df.loc[df.index.weekday < 5, 'period'] = 'workday'  # 工作日

        # 将特征转换为数值
        df['is_major_peak'] = (df['period'] == 'major_peak').astype(int)
        df['is_minor_peak'] = (df['period'] == 'minor_peak').astype(int)
        df['is_holiday'] = (df['period'] == 'holiday').astype(int)
        df['is_workday'] = (df['period'] == 'workday').astype(int)

        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[['pickups', 'delivers', 'is_major_peak', 'is_minor_peak', 'is_holiday', 'is_workday']])

        # 更新 create_dataset 函数
        def create_dataset(dataset, time_step=1):
            dataX = [dataset[i:(i + time_step), :] for i in range(len(dataset) - time_step - 1)]
            dataY = [dataset[i + time_step, :2] for i in range(len(dataset) - time_step - 1)]
            return np.array(dataX), np.array(dataY)
        
        # 添加步长和数据集比例的输入组件
        time_step = st.slider('选择时间步长', min_value=1, max_value=60, value=30, step=1)
        train_ratio = st.slider('选择训练集比例', min_value=0.1, max_value=0.9, value=0.6, step=0.05)
        val_ratio = st.slider('选择验证集比例', min_value=0.05, max_value=0.3, value=0.2, step=0.05)

        # 输入预测天数
        num_days = st.number_input('输入预测天数', min_value=1, value=6, step=1)

        # 计算数据集大小
        train_size = int(len(scaled_data) * train_ratio)
        val_size = int(len(scaled_data) * val_ratio)
        test_size = len(scaled_data) - train_size - val_size
        
        '''# 重新创建训练集、验证集和测试集
        time_step = 30
        train_size = int(len(scaled_data) * 0.6)  # 训练集占60%
        val_size = int(len(scaled_data) * 0.2)    # 验证集占20%
        test_size = len(scaled_data) - train_size - val_size  # 测试集占20%'''

        train_data = scaled_data[0:train_size, :]
        val_data = scaled_data[train_size:train_size + val_size, :]
        test_data = scaled_data[train_size + val_size:len(scaled_data), :]

        # 调整输入数据的形状以适应LSTM
        X_train = create_dataset(train_data, time_step)[0]
        X_val = create_dataset(val_data, time_step)[0]
        X_test = create_dataset(test_data, time_step)[0]

        y_train = create_dataset(train_data, time_step)[1]
        y_val = create_dataset(val_data, time_step)[1]
        y_test = create_dataset(test_data, time_step)[1]

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 6)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 6)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 6)

        # 重新构建LSTM模型
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 6)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(2))

        # 编译模型
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 使用早停
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        epochs_times = st.slider('选择训练次数', min_value=50, max_value=300, value=100, step=1)

        # 训练按钮
        if st.button('开始训练'):

            # 训练模型
            history = model.fit(X_train, y_train, epochs=epochs_times, batch_size=64, validation_data=(X_val, y_val))
            
            # 保存模型
            joblib.dump(model, 'model.pkl')

            # 预测
            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)

            # 反归一化
            train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], 4))), axis=1))[:, :2]
            test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], 4))), axis=1))[:, :2]

            y_train = scaler.inverse_transform(np.concatenate((y_train, np.zeros((y_train.shape[0], 4))), axis=1))[:, :2]
            y_test = scaler.inverse_transform(np.concatenate((y_test, np.zeros((y_test.shape[0], 4))), axis=1))[:, :2]

            # 获取最新的实际数据
            last_known_data = df[['pickups', 'delivers', 'is_major_peak', 'is_minor_peak', 'is_holiday', 'is_workday']].iloc[-1]

            # 定义 test_indices
            test_indices = df.index[train_size + val_size + time_step + 1:]

            def dynamic_predict_future(model, df, time_step, num_days, scaler):
                # 初始化未来预测数据
                future_data = df[['pickups', 'delivers', 'is_major_peak', 'is_minor_peak', 'is_holiday', 'is_workday']].values[-time_step:]
                future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=num_days, freq='D')
                
                # 生成未来预测值
                future_predict = []
                for i in range(num_days):
                    # 归一化未来数据
                    scaled_future_data = scaler.transform(future_data)
                    # 调整形状以适应LSTM
                    scaled_future_data = scaled_future_data.reshape(1, time_step, 6)
                    # 进行预测
                    next_day_predict = model.predict(scaled_future_data)
                    # 反归一化
                    next_day_predict = scaler.inverse_transform(np.concatenate((next_day_predict, np.zeros((next_day_predict.shape[0], 4))), axis=1))[:, :2]
                    
                    # 创建一个新的预测数据行，包含预测的 pick-ups 和 delivers 以及当前的 is_major_peak, is_minor_peak, is_holiday 和 is_workday
                    new_date = future_dates[i]
                    is_major_peak = 1 if new_date in major_peak_dates else 0
                    is_minor_peak = 1 if new_date in minor_peak_dates else 0
                    is_holiday = 1 if new_date in holiday_dates else 0
                    is_workday = 1 if new_date.weekday() < 5 and not is_holiday else 0
                    
                    new_row = np.concatenate((next_day_predict[0], [is_major_peak, is_minor_peak, is_holiday, is_workday]))
                    
                    # 更新未来数据
                    future_data = np.append(future_data[1:], [new_row], axis=0)
                    
                    # 将预测结果添加到 DataFrame 中
                    new_data = {
                        'pickups': next_day_predict[0][0],
                        'delivers': next_day_predict[0][1],
                        'is_major_peak': is_major_peak,
                        'is_minor_peak': is_minor_peak,
                        'is_holiday': is_holiday,
                        'is_workday': is_workday,
                        'period': 'normal' if is_workday else ('holiday' if is_holiday else 'weekend')
                    }
                    new_df = pd.DataFrame(new_data, index=[new_date])
                    df = pd.concat([df, new_df])
                    
                    # 保存预测结果
                    future_predict.append(next_day_predict[0])
                
                # 将预测结果转换为NumPy数组
                future_predict = np.array(future_predict)
                return future_predict, future_dates, df

            # Streamlit 应用
            st.title('收派件量预测')



            future_predict, future_dates, df_extended = dynamic_predict_future(model, df, time_step, num_days, scaler)
            
            # 打印 future_predict 的形状和内容
            st.write("future_predict shape:", future_predict.shape)
            st.write("future_predict content:", future_predict)
            
            # 打印 future_dates 以确认日期范围
            st.write("future_dates:", future_dates)

            # 绘制训练和验证损失曲线
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(history.history['loss'], label='训练损失')
            ax1.plot(history.history['val_loss'], label='验证损失')
            ax1.set_title('训练和验证损失')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            st.pyplot(fig1)

            # 收件量预测
            fig2, ax2 = plt.subplots(2, 1, figsize=(12, 12))
            ax2[0].plot(df.index[time_step+1:train_size], y_train[:, 0], label='实际收件量')
            ax2[0].plot(df.index[time_step+1:train_size], train_predict[:, 0], label='预测收件量', color='red')
            if test_predict.size > 0:
                ax2[0].plot(test_indices, y_test[:, 0], label='实际收件量 (测试集)')
                ax2[0].plot(test_indices, test_predict[:, 0], label='预测派件量 (测试集)', color='orange')
            ax2[0].plot(future_dates, future_predict[:, 0], label='未来预测收件量', color='green')
            ax2[0].set_title('收件量预测')
            ax2[0].legend()

            # 派件量预测
            ax2[1].plot(df.index[time_step+1:train_size], y_train[:, 1], label='实际派件量')
            ax2[1].plot(df.index[time_step+1:train_size], train_predict[:, 1], label='预测派件量', color='red')
            if test_predict.size > 0:
                ax2[1].plot(test_indices, y_test[:, 1], label='实际派件量 (测试集)')
                ax2[1].plot(test_indices, test_predict[:, 1], label='预测派件量 (测试集)', color='orange')
            ax2[1].plot(future_dates, future_predict[:, 1], label='未来预测派件量', color='green')
            ax2[1].set_title('派件量预测')
            ax2[1].legend()

            # 显示图表
            st.pyplot(fig2)

            # 显示未来预测的数据
            future_df = pd.DataFrame(future_predict, columns=['预测收件量', '预测派件量'], index=future_dates)
            st.write("未来预测数据:")
            st.dataframe(future_df)

            # 下载未来预测数据
            csv = future_df.to_csv(index_label='日期')
            st.download_button(
                label="下载未来预测数据",
                data=csv,
                file_name="future_predictions.csv",
                mime="text/csv"
            )
