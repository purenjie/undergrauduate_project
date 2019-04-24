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

采用小波去噪，采用 4 层去噪，软阈值处理，阈值处理的小波系数层数分别为 1 和 4。

3. 归一化处理

若不进行归一化处理，支持向量机的 mse（均方误差）会较大。

### 代码编写

- 调用库

```python
import pandas as pd  # 读取 excel 文件并对数据进行操作
from sklearn.svm import SVR  # 选择核函数和调整参数
import matplotlib.pyplot as plt  # 画图
import numpy as np  # 数组操作
from sklearn.model_selection import train_test_split  # 将数据划分为训练集和测试集
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


# 小波去噪
# lv为分解层数；data为最后保存的dataframe便于作图；
# index_list为待处理序列；wavefunc为选取的小波函数；
# m,n则选择了进行阈值处理的小波系数层数
def wt(index_list,wavefunc,lv,m,n):    
    # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn函数

    # 去噪过程
    for i in range(m,n+1):   # 选取小波系数层数为 m~n层，尺度系数不需要处理
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2*np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 向零收缩
            else:
                coeff[i][j] = 0   # 低于阈值置零
    # 重构
    denoised_index = pywt.waverec(coeff,wavefunc)
    return denoised_index[1:]


# 归一化处理
x = preprocessing.scale(x)
y = preprocessing.scale(y)
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
# 画出模型对数据的输入输出数据的预测情况
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
# 画出训练后模型 预测值 和 实际值 的图像
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

    pre_rate = svr_model.score(x_test, y_test)
    print('决定系数：%.4f' % pre_rate)

    y_pred = svr_rbf.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print('均方误差：%.4f' % mse) 

    log_file = '/home/solejay/program/undergrauduate_project/log.txt'
    with open(log_file, 'a') as f:
        s0 = '预测率：%.4f' % pre_rate + '\n'
        s1 = '均方误差：%.4f' % mse + '\n'
        s2 = '读取文件：' + excel_file.split('/')[-1] + '\n'
        s3 = '模型参数：' + str(svr_model) + '\n'
        s4 = '=============================================================\n'
        s = s0 + s1 + s2 + s3 + s4
        f.write(s)
```

### [完整代码](https://github.com/purenjie/undergrauduate_project/blob/master/code/svr.py)



