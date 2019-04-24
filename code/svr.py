import pywt
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

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


# 画出预测值和实际值的图像
def plot_graph(svr_model, x_test, y_test):
    sample = [i for i in range(1, len(y_test)+1)]
    sample = np.reshape(sample, (len(sample), 1))
    
    y_pred = svr_rbf.predict(x_test)

    plt.plot(sample, y_test, color='black', label='y_test')
    plt.plot(sample, y_pred, color='red', label='y_pred')

    plt.xlabel('sample')
    plt.ylabel('utilization')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    excel_file = '/home/solejay/program/undergrauduate_project/excel/all.xlsx'
    data = get_data(excel_file)

    x = data.iloc[:, 0:5]
    y = data.iloc[:, 5]

    # 异常值处理
    x, y = remove_outlier(x, y)

    # 小波去噪
    for i in range(5):
        x.iloc[:, i] = wt(x.iloc[:, i],'db4',4,1,4) 
    y = wt(y,'db4',4,1,4) 

    # 归一化处理
    x = preprocessing.scale(x)
    y = preprocessing.scale(y)

    # 划分训练集和测试集
    # x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:5], data.iloc[:, 5], test_size=0.25, random_state=33)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

    # 核函数
    svr_rbf = SVR(kernel='rbf', C=1, gamma='auto')
    svr_rbf.fit(x_train, y_train)

    write_log(svr_rbf, x_test, y_test, excel_file)
    plot_graph(svr_rbf, x_test, y_test)



