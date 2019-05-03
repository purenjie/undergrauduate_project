import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet

# 异常值处理：——拉依达法则：数据偏差大于三倍标准差剔除
def remove_outlier(data):
    y = data.iloc[:, 6]
    mean = np.mean(y)  # 平均值
    std = np.std(y)  # 标准差
    
    lower_limit = mean-3*std  # 最小值
    upper_limit = mean+3*std  # 最大值
    
    for i in range(y.shape[0]):
        if y[i]<lower_limit or y[i]>upper_limit:
            data = data.drop(i)
    return data

def plot_utilization(y, y1):
    sample = [i for i in range(1, len(y)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    
    plt.plot(sample, y, color='black', label='before')
    plt.plot(sample, y1, color='red', label='after')
    
    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


excel_file = '/home/solejay/program/undergrauduate_project/excel/1000.xlsx'
data = get_data(excel_file)

x = data.iloc[:, 0:6] 
y = data.iloc[:, 6] 
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0, random_state=0)

excel_file1 = '/home/solejay/program/undergrauduate_project/excel/去噪1000.xlsx'
data1 = get_data(excel_file1)

x1 = data.iloc[:, 0:6] 
y1 = data.iloc[:, 6] 
x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0, random_state=0)

# x1 = data.iloc[:, 0]  # 风量
# x2 = data.iloc[:, 1]  # 风压
# x3 = data.iloc[:, 2]  # 顶压
# y = data.iloc[:, 6] 

# x11 = data1.iloc[:, 0]
# x22 = data1.iloc[:, 1]
# x33 = data1.iloc[:, 2]
# y1 = data1.iloc[:, 6] 

plot_utilization(y_train, y_train1)
# print(len(y))
# print(len(y1))

# writer = pd.ExcelWriter('output.xlsx')
# output.to_excel(writer,'Sheet1')
# writer.save()