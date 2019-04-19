import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 读取 excel 文件，默认返回第一张表 
# 返回类型：<class 'pandas.core.frame.DataFrame'>
def get_data(file):
    sheet = pd.read_excel(io=file)
    return sheet

# 输出模型预测率 并写入日志文件
def write_log(svr_model, x_test, y_test, excel_file):

    pre_rate = svr_model.score(x_test, y_test)
    print('预测率：%.4f' % pre_rate)

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

    excel_file = '/home/solejay/program/undergrauduate_project/excel/more.xlsx'
    data = get_data(excel_file)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:5], data.iloc[:, 5], test_size=0.25, random_state=33)

    # 核函数
    svr_rbf = SVR(kernel='rbf', gamma='auto')
    svr_rbf.fit(x_train, y_train)

    write_log(svr_rbf, x_test, y_test, excel_file)
    plot_graph(svr_rbf, x_test, y_test)



