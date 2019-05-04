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
    return(pywt.waverec(coeff,wavefunc))

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

# 画出预测值和实际值的图像
def plot_graph(svr_model, x_test, y_test):
    sample = [i for i in range(1, len(y_test)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    
    y_pred = svr_model.predict(x_test)

    plt.plot(sample, y_test, color='black', label='真实值')
    plt.plot(sample, y_pred, color='black', marker='p', label='预测值')

    plt.xlabel('样本')
    plt.ylabel('CO利用率 %')
    plt.title('BP 神经网络')
    plt.legend()
    plt.show()

excel_file = '/home/solejay/program/undergrauduate_project/excel/1000.xlsx'
data = get_data(excel_file)

x = data.iloc[:, 0:6]
y = data.iloc[:, 6] 

# 异常值处理
x, y = remove_outlier(x, y) # 1000 组剔除 4 组数据   1000.xlsx

for i in range(5):
    x.iloc[:, i] = wt(x.iloc[:, i],'db5',4,1,4) 
y = wt(y,'db5',4,1,4) 


# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# 归一化处理
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(x_train)
x_test = x_scaler.transform(x_test)


# parameters = {'kernel':['rbf'], 'gamma':np.logspace(-5, 3, num=6, base=2),'C':np.logspace(-2, 3, num=5)}
# grid_search = GridSearchCV(SVR(), parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')
# grid_search.fit(x_train,y_train)

# write_log(grid_search, x_test, y_test, excel_file)
# plot_graph(grid_search, x_test, y_test)

clf = MLPRegressor(solver='lbfgs',random_state=0)
clf.fit(x_train, y_train)

write_log(clf, x_test, y_test, excel_file)
plot_graph(clf, x_test, y_test)

svr_rbf = SVR(kernel='rbf', gamma='auto')
svr_rbf.fit(x_train, y_train)

write_log(svr_rbf, x_test, y_test, excel_file)
plot_graph(svr_rbf, x_test, y_test)