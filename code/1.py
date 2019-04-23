import pywt
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet


def plot_graph(x, y, svr_model):
    # 样本数——横轴
    sample = [i for i in range(1, len(x)+1)]
    sample = np.reshape(sample, (len(sample), 1))

    # 根据给定的训练数据拟合SVM模型
    svr_model.fit(x, y)

    
    plt.plot(sample, y, color='black', label='Data')  # 实际数据
#     plt.plot(sample, svr_model.predict(x), color='red', label='RBF model')  # 预测数据

    plt.xlabel('sample')  # x 轴标签
    plt.ylabel('utilization')  # y 轴标签
    plt.title('Support Vector Regression')  # 图像标题
    plt.legend()  # 显示图例（label）
    plt.show()  # 显示图像

def plot_graph_1(x, y):
    plt.plot(np.arange(1, len(x)+1).reshape((len(x), 1)), x, color='black', label='Data')
    plt.plot(np.arange(1, len(y)+1).reshape((len(y), 1)), y, color='red', label='Data')
    plt.show()

# 打包为函数，方便调节参数。  
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
    return denoised_index

excel_file = '/home/solejay/program/undergrauduate_project/excel/more.xlsx'
data = get_data(excel_file)

x = data.iloc[:, 0:5]
y = data.iloc[:, 5]
svr_rbf = SVR(kernel='rbf', gamma='auto')

x1 = x.iloc[:, 0]  # 风压
x2 = x.iloc[:, 1]  # 顶压
x3 = x.iloc[:, 2]  # 风温
x4 = x.iloc[:, 3]  # O2_FYL
x5 = x.iloc[:, 4]  # 顶温

x1_ = wt(x1,'db4',4,2,4) 
x2_ = wt(x2,'db4',4,2,4) 
x3_ = wt(x3,'db4',4,2,4) 
x4_ = wt(x4,'db4',4,2,4) 
x5_ = wt(x5,'db4',4,2,4) 
y_ = wt(y,'db4',4,1,4) 

plot_graph_1(x1, x1_)
plot_graph_1(x2, x2_)
plot_graph_1(x3, x3_)
plot_graph_1(x4, x4_)
plot_graph_1(x5, x5_)
plot_graph_1(y, y_)

