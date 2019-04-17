import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(file):

    sheet = pd.read_excel(io=file)
    return sheet


def plot_graph(x, y):
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))

    # 核函数
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models
    svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
    svr_rbf.fit(x, y)  # fitting the data points in the models
    svr_lin.fit(x, y)
    svr_poly.fit(x, y)  # poly 核函数显示不出图像

    # 散点图——实际数据
    plt.scatter(sample, y, color='black', label='Data')

    # 实线图——预测数据
    plt.plot(sample, svr_rbf.predict(x), color='red', label='RBF model')
    plt.plot(sample, svr_lin.predict(x), color='blue', label='Linear model')
    plt.plot(sample, svr_poly.predict(x), color='yellow', label='Poly model')
    
    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

file = '/home/solejay/program/bishe/learn/test.xlsx'

data = get_data(file)
# print(data) 9行数据
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:11], data.iloc[:, 11], test_size=0.25, random_state=33)

x = data.iloc[:, 0:11]
y = data.iloc[:, 11]


plot_graph(x, y)

