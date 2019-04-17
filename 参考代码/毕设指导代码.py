import numpy as np
from sklearn.metrics import roc_curve, auc,roc_auc_score  ###计算roc和auc
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
#读取文件
raw_data = pd.read_csv('stock.csv')
data = raw_data[raw_data['stock_code']=='603912.SH']
#选择特征
data = data.loc[0:100,['open','close','high','low','amt_change','pct_change','amount','vol']]
#data.dropna(inplace=True)
#读取开盘、收盘价，设置股票涨跌标签
data_open=raw_data['open']
data_close=raw_data['close']
y=[]
num_x=len(data)
for i in range(num_x):
    if data_open[i]>=data_close[i]:
        y.append(1)
    else:
        y.append(0)
x_data=data.as_matrix()
x=x_data                             #到这里x和y都已经准备好了
data_shape=x.shape
data_rows=data_shape[0]
data_cols=data_shape[1]
data_col_max=x.max(axis=0)
data_col_min=x.min(axis=0)
#将输入数组归一化
for i in range(0, data_rows, 1):
    for j in range(0, data_cols, 1):
        x[i][j] = (x[i][j] - data_col_min[j]) / (data_col_max[j] - data_col_min[j])
#训练模型
clf1 = svm.SVC(kernel='rbf')
# x和y的验证集和测试集，3：1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 训练数据进行训练
clf1.fit(x_train, y_train)
#计算正确率
y_pre_test = clf1.predict(x_test)
result = np.mean(y_test == y_pre_test)
#模型评估
print('svm classifier accuacy = %.2f'%(result))
y_test = np.array(y_test)
y_pre_test = np.array(y_pre_test)
print(' AUC = %.2f'%(roc_auc_score(y_test, y_pre_test)))
fpr,tpr,threshold = roc_curve(y_test, y_pre_test)              #计算真正率和假正率
roc_auc = auc(fpr,tpr)                                         #计算auc的值
#绘制roc曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
