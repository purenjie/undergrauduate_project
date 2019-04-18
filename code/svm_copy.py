import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 读取 excel 文件，默认返回第一张表
def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet

# 输出模型预测率
def score(svr_model, x, y):
    svr_model.fit(x, y)
    print('预测率：', svr_model.score(x, y))

# 传入  输入 输出 核函数 
# 画图  输出 预测输出
def plot_graph(x, y, svr_model):
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    svr_model.fit(x, y)

    # 实线图——实际数据
    plt.plot(sample, y, color='black', label='Data')

    # 实线图——预测数据
    plt.plot(sample, svr_model.predict(x), color='red', label='RBF model')

    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

# 测试集 实际输出 和 预测输出
def plot_result(y_test, y_pre_test):
    sample = [i for i in range(1, len(y_test)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    
    plt.scatter(sample, y_test, color='black', label='y_test')
    plt.plot(sample, y_pre_test, color='red', label='y_pre_test')

    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

file = '/home/solejay/program/bishe/learn/less.xlsx'
data = get_data(file)
# print(type(data))  # <class 'pandas.core.frame.DataFrame'>
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:5], data.iloc[:, 5], test_size=0.25, random_state=33)

x = data.iloc[:, 0:5]
y = data.iloc[:, 5]
# print(x)
# print(y)
# print(type(x))  # <class 'pandas.core.frame.DataFrame'>
# print(type(y))  # <class 'pandas.core.series.Series'>

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
y_pre_test = svr_rbf.predict(x_test)
# print(type(np.array(y_pre_test)))  # <class 'numpy.ndarray'>
# print(type(np.array(y_test)))  # <class 'numpy.ndarray'>

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_val = rmse(np.array(y_pre_test), np.array(y_test))
print('均方差：', rmse_val)





# plot_graph(x, y, svr_rbf)
# score(svr_rbf, x, y)
# score(svr_rbf, x_train, y_train)
# score(svr_rbf, x_test, y_test)
# plot_result(y_test, y_pre_test)

