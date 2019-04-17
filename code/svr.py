import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(file):

    sheet = pd.read_excel(io=file)
    return sheet

def plot_graph(x, y, svr_model):
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    svr_model.fit(x, y)

    # 散点图——实际数据
    plt.scatter(sample, y, color='black', label='Data')
    # 实线图——预测数据
    plt.plot(sample, svr_model.predict(x), color='red', label='RBF model')
    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

file = '/home/solejay/program/bishe/learn/备用数据.xlsx'

data = get_data(file)
# print(data) 9行数据
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:11], data.iloc[:, 11], test_size=0.25, random_state=33)

x = data.iloc[:, 0:11]
y = data.iloc[:, 11]

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

plot_graph(x, y, svr_rbf)

