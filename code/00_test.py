file = '/home/solejay/program/bishe/learn/more.xlsx'
data = get_data(file)
# print(type(data))  # <class 'pandas.core.frame.DataFrame'>

# 整体集
x = data.iloc[:, 0:5]
y = data.iloc[:, 5]
# print(x)
# print(y)
# print(type(x))  # <class 'pandas.core.frame.DataFrame'>
# print(type(y))  # <class 'pandas.core.series.Series'>

# 训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:5], data.iloc[:, 5], test_size=0.25, random_state=33)

# 核函数
svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')
svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')

# svr 核函数整体集图像
# plot_graph(x, y, svr_rbf)

#　输出整体集、训练集、测试集的预测准确率
score(svr_rbf, x, y)
score(svr_rbf, x_train, y_train)
score(svr_rbf, x_test, y_test)


# 测试集预测结果图
svr_rbf.fit(x_test, y_test)
y_pre_test = svr_rbf.predict(x_test)
plot_result(y_test, y_pre_test)

# 整体集、训练集、测试集图像
# plot_three_kernel(x, y)
# plot_three_kernel(x_train, y_train)
# plot_three_kernel(x_test, y_test)