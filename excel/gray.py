import numpy as np
import pandas as pd

# 读取为df格式
gray = pd.read_excel(
    "/home/solejay/program/undergrauduate_project/excel/原始数据.xlsx")

# min-max标准化
gray = (gray - gray.min()) / (gray.max() - gray.min())

std = gray.iloc[:, 11]  # 为标准要素
cmp = gray.iloc[:, 0:11]  # 为比较要素

# cmp.shape  tuple (2999, 11)
row = cmp.shape[0]  # 行数
col = cmp.shape[1]  # 列数

# 与标准要素比较，相减
a = np.zeros([col, row])  # 11*2999 矩阵
for i in range(col):
    for j in range(row):
        a[i, j] = abs(cmp.iloc[j, i]-std[j])

# 取出矩阵中最大值与最小值
c = np.amax(a)
d = np.amin(a)

# 计算关联系数
result = np.zeros([col, row])
for i in range(col):
    for j in range(row):
        result[i, j] = (d+0.5*c)/(a[i, j]+0.5*c)

# 求均值，得到灰色关联值
result2 = np.zeros(col)
for i in range(col):
        result2[i] = np.mean(result[i, :])
RT = pd.DataFrame(result2)
RT.to_csv("/home/solejay/program/undergrauduate_project/excel/灰色关联1out.csv")
