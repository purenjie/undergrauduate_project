import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pywt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

# 读取 excel 文件，默认返回第一张表 
# 返回类型：<class 'pandas.core.frame.DataFrame'>
def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet

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

# 打包为函数，方便调节参数。  
# lv为分解层数；data为最后保存的dataframe便于作图；
# index_list为待处理序列；wavefunc为选取的小波函数；
# m,n则选择了进行阈值处理的小波系数层数
# 函数打包
def wt(index_list, wavefunc,lv,m,n):   # 打包为函数，方便调节参数。  lv为分解层数；data为最后保存的dataframe便于作图；index_list为待处理序列；wavefunc为选取的小波函数；m,n则选择了进行阈值处理的小波系数层数
    # 分解
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   # 按 level 层分解，使用pywt包进行计算， cAn是尺度系数 cDn为小波系数

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
    return(pywt.waverec(coeff,wavefunc)[1:])

# 输出模型预测率 并写入日志文件
def accuracy(svr_model, x_test, y_test):
    
    y_pred = svr_model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print('均方误差：%.4f' % mse) 
    
    mae = mean_absolute_error(y_test, y_pred)
    print('平均误差：%.4f' % mae)
    
    return (mse, mae)


excel_file = '/home/solejay/program/undergrauduate_project/excel/全部1000.xlsx'
data = get_data(excel_file)

x = data.iloc[:, 0:9]
y = data.iloc[:, 10] 

# 异常值处理
x, y = remove_outlier(x, y) # 1000 组剔除 4 组数据   1000.xlsx

for i in range(9):
    x.iloc[:, i] = wt(x.iloc[:, i],'db5',4,1,4) 
y = wt(y,'db5',4,1,4) 


# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 标准化处理
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)


# clf = MLPRegressor(solver='lbfgs', random_state=0, max_iter=1000)
# clf.fit(x_train, y_train)

parameters = {'kernel':['rbf'], 'gamma':np.logspace(-5, 3, num=6, base=2),'C':np.logspace(-2, 3, num=5)}
grid_search = GridSearchCV(SVR(), parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')
grid_search.fit(x_train,y_train)

MSE = []
MAE = []

for i in range(30):
    mse, mae = accuracy(grid_search, x_test, y_test)
    MSE.append(mse)
    MAE.append(mae)

mse_mean = np.mean(MSE)
mae_mean = np.mean(MAE)

print(mse_mean)
print(mae_mean)


# parameters = {'kernel':['rbf'], 'gamma':np.logspace(-5, 3, num=6, base=2),'C':np.logspace(-2, 3, num=5)}
# grid_search = GridSearchCV(SVR(), parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')
# grid_search.fit(x_train,y_train)

# write_log(grid_search, x_test, y_test, excel_file)
# plot_graph(grid_search, x_test, y_test)


# svr_rbf = SVR(kernel='rbf', gamma='auto')
# svr_rbf.fit(x_train, y_train)

# write_log(svr_rbf, x_test, y_test, excel_file)
# plot_graph(svr_rbf, x_test, y_test)