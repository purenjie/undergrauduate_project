# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
from sklearn import metrics

# 读取 excel 文件，默认返回第一张表 
# 返回类型：<class 'pandas.core.frame.DataFrame'>
def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet

excel_file = '/home/solejay/program/undergrauduate_project/excel/less.xlsx'
data = get_data(excel_file)

x = data.iloc[:, 0:6]
y = data.iloc[:, 6]

parameters = {'kernel':['rbf'], 'gamma':np.logspace(-5, 0, num=6, base=2.0),'C':np.logspace(-5, 5, num=11, base=2.0)}
grid_search = GridSearchCV(svm.SVR(), parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# # 打乱训练数据——本来就是随机分的训练集和测试集，没什么用
# random_seed = 13
# X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)

# 标准化
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

grid_search.fit(X_train,y_train)
y_pred = grid_search.predict(X_test) 

print('mean_squared_error:'+str(metrics.mean_squared_error(y_test,y_pred)),\
 'r2_score:'+str(metrics.r2_score(y_test,y_pred)))