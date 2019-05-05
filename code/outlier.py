import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet

def remove_outlier(x, y):
    mean = np.mean(y)  # 平均值
    std = np.std(y)  # 标准差
    
    lower_limit = mean-3*std
    upper_limit = mean+3*std
    
    for i in range(y.shape[0]):
        if y[i]<lower_limit or y[i]>upper_limit:
            x = x.drop(i)
            y = y.drop(i)
    return x, y

def plot_graph(y, y_after):
    sample = [i for i in range(1, len(y)+1)]
    sample = np.reshape(sample, (len(sample), 1))

    sample_after = [i for i in range(1, len(y_after)+1)]
    sample_after = np.reshape(sample_after, (len(sample_after), 1))

    # plt.plot(sample, y, color='black', label='异常值处理前')
    plt.plot(sample_after, y_after, color='black', label='异常值处理后')

    plt.xlabel('样本')
    plt.ylabel('CO利用率 %')
    plt.title('异常值处理')
    plt.legend()
    plt.show()

excel_file = '/home/solejay/program/undergrauduate_project/excel/全部1000.xlsx'
data = get_data(excel_file)
x = data.iloc[:, 0:9]
y = data.iloc[:, 10]

x_after, y_after = remove_outlier(x, y)  

plot_graph(y, y_after)