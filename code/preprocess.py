import numpy as np
import pandas as pd
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


excel_file = '/home/solejay/program/undergrauduate_project/excel/all.xlsx'
data = get_data(excel_file)

print(len(data))

# 异常值处理
output = remove_outlier(data) # 1000 组剔除 4 组数据   1000.xlsx

print(len(output))


writer = pd.ExcelWriter('output.xlsx')
output.to_excel(writer,'Sheet1')
writer.save()