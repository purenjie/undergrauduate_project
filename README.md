# undergrauduate_project

### 毕设任务

运用支持向量机回归（SVR），预测一氧化碳利用率

### 数据预处理

### 输入数据

- 风压（blast pressure）
- 顶压（top pressure）
- 风温（blast temperature）
- 氧增强比（OER）
- 顶温（top temperature）

### 输出数据

- 一氧化碳利用率

### 预处理流程

因为所给数据大致经过处理，没有缺失值的情况。因此没有缺失值的处理。

1. 异常值处理

采用拉依达准则法对异常数据进行多次反复剔除。

样本值和平均值的查的绝对值大于三倍的标准差时，将该样本值剔除。

2. 数据平滑处理（去噪）
3. 标准化处理

新数据=（原数据-均值）/标准差

z-score标准化，也称为标准化分数，这种方法根据原始数据的均值和标准差进行标准化，经过处理后的数据符合标准正态分布，即均值为0，标准差为1（根据下面的转化函数很容易证明），转化函数为：

![](https://img-blog.csdn.net/20160706160124006?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

所以说，这种标准化我们称之为归一化的时候，本质上是指将原始数据的标准差映射成1，是标准差归一化。标准差分数可以回答这样一个问题：“给定数据距离其均值多少个标准差”的问题，在均值之上的数据会得到一个正的标准化分数，反之会得到一个负的标准化分数。

### 代码编写

- 调用库

```python
import pandas as pd  # 读取 excel 文件并对数据进行操作
from sklearn.svm import SVR  # 选择核函数和调整参数
import matplotlib.pyplot as plt  # 画图
import numpy as np  # 数组操作
from sklearn.model_selection import train_test_split  # 将数据划分为训练集和测试集
from sklearn.metrics import mean_squared_error  # 计算 mse
from sklearn.metrics import mean_absolute_error  # 计算 mae
from sklearn.model_selection import GridSearchCV  # 交叉验证选取最佳 SVR 系数
from sklearn.preprocessing import StandardScaler  # z-score 标准化
```

- 数据预处理

```python
# 异常值处理：——拉依达法则：数据偏差大于三倍标准差剔除
def remove_outlier(x, y):
    mean = np.mean(y)  # 平均值
    std = np.std(y)  # 标准差
    
    lower_limit = mean-3*std  # 最小值
    upper_limit = mean+3*std  # 最大值
    
    for i in range(y.shape[0]):
        if y[i]<lower_limit or y[i]>upper_limit:
            x = x.drop(i)
            y = y.drop(i)
    return x, y

# 归一化处理
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)
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
# 画出预测值和实际值的图像
def plot_graph(svr_model, x_test, y_test):
    # 样本数——横轴
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))

    y_pred = svr_model.predict(x_test)

    plt.plot(sample, y_test, color='black', label='y_test')
    plt.plot(sample, y_pred, color='red', label='y_pred')

    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
```

```python
# 画出三种不同核函数的拟合图像
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

- 数据记录

```python
# 输出模型预测率 并写入日志文件
def write_log(svr_model, x_test, y_test, excel_file):
    
    y_pred = svr_model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print('均方误差：%.4f' % mse) 
    
    mae = mean_absolute_error(y_test, y_pred)
    print('平均误差：%.4f' % mae)

    log_file = '/home/solejay/program/undergrauduate_project/log1.txt'
    with open(log_file, 'a') as f:
        s1 = '均方误差：%.4f' % mse + '\n'
        s2 = '平均误差：%.4f' % mae + '\n'
        s3 = '读取文件：' + excel_file.split('/')[-1] + '\n'
        s4 = '模型参数：' + str(svr_model) + '\n'
        s5 = '=============================================================\n'
        s = s1 + s2 + s3 + s4 + s5
        f.write(s)
```

### [完整代码](https://github.com/purenjie/undergrauduate_project/blob/master/code/svr.py)



