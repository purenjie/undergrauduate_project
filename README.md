# undergrauduate_project

### 毕设任务

运用支持向量机回归（SVR），预测一氧化碳利用率

### 数据预处理

### 输出数据

- 风压（blast pressure）
- 顶压（top pressure）
- 风温（blast temperature）
- 氧增强比（OER）
- 顶温（top temperature）

### 输出数据

- 一氧化碳利用率

### 代码编写

- 调用库

```python
import pandas as pd  # 读取 excel 文件并对数据进行操作
from sklearn.svm import SVR  # 选择核函数和调整参数
import matplotlib.pyplot as plt  # 画图
import numpy as np  # 数组操作
from sklearn.model_selection import train_test_split  # 将数据划分为训练集和测试集
```

- 定义函数

```python
# 读取 excel 文件，默认返回第一张表 
# 返回类型：<class 'pandas.core.frame.DataFrame'>
def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet
```

```python
# 输出模型预测率
def score(svr_model, x, y):
    svr_model.fit(x, y)
    print('预测率：', svr_model.score(x, y))
```

```python
# 传入  输入 输出 核函数 
# 画图  输出 预测输出
def plot_graph(x, y, svr_model):
    # 样本数——横轴
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))

    # 根据给定的训练数据拟合SVM模型
    svr_model.fit(x, y)

    
    plt.plot(sample, y, color='black', label='Data')  # 实际数据
    plt.plot(sample, svr_model.predict(x), color='red', label='RBF model')  # 预测数据

    plt.xlabel('sample')  # x 轴标签
    plt.ylabel('utilization')  # y 轴标签
    plt.title('Support Vector Regression')  # 图像标题
    plt.legend()  # 显示图例（label）
    plt.show()  # 显示图像
```

```python
# 测试集 实际输出 和 预测输出
def plot_result(y_test, y_pre_test):
    sample = [i for i in range(1, len(y_test)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    
    plt.plot(sample, y_test, color='black', label='y_test')
    plt.plot(sample, y_pre_test, color='red', label='y_pre_test')

    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
```

```python
# 画出三种不同和函数的拟合图像
def plot_three_kernel(x, y):
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))

    # 核函数
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models
    svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')  # 效果太差

    # 拟合输入数据
    svr_rbf.fit(x, y)  # fitting the data points in the models
    svr_lin.fit(x, y)
    # svr_poly.fit(x, y)  # poly 核函数显示不出图像

    # 散点图——实际数据
    plt.scatter(sample, y, color='black', label='Data')

    # 实线图——预测数据
    plt.plot(sample, svr_rbf.predict(x), color='red', label='RBF model')
    plt.plot(sample, svr_lin.predict(x), color='blue', label='Linear model')
    # plt.plot(sample, svr_poly.predict(x), color='yellow', label='Poly model')
    
    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
```



