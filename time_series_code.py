
# ================导入基础包================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ================读取数据，并且进行初步画图================
# 海平面数据 2010-2019
path = r"D:\lagua\study\ML\time_series\NH.Ts+dSST.xlsx" # 路径
init_data = pd.read_excel(path, sheet_name='NH.Ts+dSST', header=1, index_col=0).loc[1990:2019, 'Jan':'Dec'] # 读取数据
y = init_data.values.ravel() # 将数据站卡
y = np.asarray(y, dtype=float)
df = pd.DataFrame(y)

# 画出数据的时序图、自相关图和偏自相关图
def myplot(data:pd.DataFrame, i:int):
    # data: pd.DataFrame
    # i: 代表第几个图
    fig, ax = plt.subplots(3, 1, num=i, figsize=(12, 21))
    ax[0].plot(data)
    lags = 2 * int(np.sqrt(len(data)))
    plot_acf(data.values, ax[1], lags=lags)
    plot_pacf(data.values, ax[2], lags=lags).show()
    for i,_ in enumerate(ax):
        if i == 0:
            ax[i].set_xticks(list(range(0, len(data), 12)))
        else:
            ax[i].set_xticks(list(range(0, lags+12, 12)))

diff_1 = df.diff(1).dropna() #一阶差分
no_season = diff_1.diff(12).dropna() # 在一阶差分的基础上进行12阶差分
diff_2 = diff_1.diff(1).dropna() # 二阶差分
# 分别画图
myplot(df, 1)
myplot(diff_1, 2)
myplot(no_season, 3)



# ================比较三种去除季节项和趋势项的方法================
# 比较三种去除趋势项和季节项的方法
# 最后得出结论，使用第一种方法，即使用一阶12步差分最好
def eliminate_trend_season(data, method, init_data=init_data):
    """
    method: 
    "pure_diff": 1阶12步差分
    "poly_diff": 多项式拟合，使用12步差分消除季节项
    "min_trend_diff": 最小趋势法去除趋势项与季节项，再在此基础上进行12步差分
    }

    传入的必须是一列数据，之后只需检查是数组，还是DataFrame
    """
    if not isinstance(data, pd.DataFrame):
        # data = np.array(data)
        data = pd.DataFrame(data)
    if method == "pure_diff":
        no_trend = data.diff(1).dropna()
        no_season = no_trend.diff(12).dropna()
        return no_season
    elif method == "poly_diff":
        # 输入的x与y
        x = list(range(len(data)))
        d_val = data.values.flatten()
        # 多项式拟合
        zzz = np.polyfit(x, d_val, 9)
        p4 = np.poly1d(zzz)
        c = p4(x)
        # 减去趋势项（多项式拟合的曲线）
        no_trend = d_val - c
        # 进行12步差分，消除季节项
        if isinstance(no_trend, pd.DataFrame):
            no_season = no_trend.diff(12).dropna()
        else:
            no_trend = pd.DataFrame(no_trend)
            no_season = no_trend.diff(12).dropna()
        return no_season
    elif method == "min_trend_diff":
        # TODO: 自动将数据转换成(10, 12)的数据框形式
        m_j = np.mean(init_data, axis=1)
        no_trend = (init_data.T - m_j).T
        no_season_3 = pd.DataFrame(no_trend.values.flatten()).diff(12).dropna()
        return no_season_3
    else:
        raise "please input a true method!!"
no_season = eliminate_trend_season(df, method="pure_diff")
myplot(no_season, 1)
# no_season = eliminate_trend_season(df, method="poly_diff")
# myplot(no_season, 2)
# no_season = eliminate_trend_season(df, method="min_trend_diff")
# myplot(no_season, 3)
# 最后得出结论，使用第一种纯差分消除趋势项和季节项最好

# ================一阶12步差分的平稳性检验================
# 平稳性检验
from statsmodels.tsa.stattools import adfuller
result = adfuller(no_season)
print(u'一阶12步差分序列的平稳性检验\n', result)
# 第二个数是e-13，这远小于0.01，所以拒绝原假设，认为时间序列是平稳的

# ================一阶差分白噪声检验================
#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'一阶差分的白噪声检验结果为：\n', acorr_ljungbox(df.diff(1).dropna(), lags=6)) #返回统计量和p值
# ================一阶12步差分白噪声检验================
#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'一阶12步序列的白噪声检验结果为：\n', acorr_ljungbox(df.diff(1).dropna().diff(12).dropna(), lags=6)) #返回统计量和p值

# ===========生成参数对，寻找最优SARIMAX()()参数===========
from itertools import product
# ARIMA的参数
ps = range(0, 1)
d = range(0, 2)
qs = range(0, 12)
# 季节项相关的参数
Ps = range(0, 1)
D = range(1, 2)
Qs = range(1, 2)
# 将参数打包，传入下面的数据，是哦那个BIC准则进行参数选择
params_list = list(product(ps, d, qs, Ps, D, Qs))
print(params_list)

# ================根据BIC 准则，寻找最优参数================
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm_notebook
from statsmodels.tsa.arima_model import ARIMA
import warnings
# 忽视在模型拟合中遇到的错误
warnings.filterwarnings("ignore")
# 找最优的参数 SARIMAX
def find_best_params(data:np.array, params_list):
    result = []
    best_bic = 100000
    for param in tqdm_notebook(params_list):
        # 模型拟合
        model = SARIMAX(data, order=(param[0], param[1], param[2]), seasonal_order=(param[3], param[4], param[5], 12)).fit(disp=-1)
        bicc = model.bic # 拟合出模型的BIC值
        # 寻找最优的参数
        if bicc < best_bic:
            best_mode = model
            best_bic = bicc
            best_param = param
        param_1 = (param[0], param[1], param[2])
        param_2 = (param[3], param[4], param[5], 12)
        param = 'SARIMA{0}x{1}'.format(param_1, param_2)
        print(param)
        result.append([param, model.bic])

    result_table = pd.DataFrame(result)
    result_table.columns = ['parameters', 'bic']
    result_table = result_table.sort_values(by='bic',ascending=True).reset_index(drop=True)
    return result_table


result_table = find_best_params(df, params_list)
print(result_table)

# ================使用上一步模型进行定阶拟合================
ma1 = SARIMAX(df, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=-1)
resid = ma1.resid
fig = ma1.plot_diagnostics(figsize=(15, 12))
fig.savefig(r'.\test.png')
# 输出模型拟合的系数，以及一切相关信息
ma1.summary()


# ================预测，并画图================
# 预测的第一个数据是0，所以去掉第一个数据
import matplotlib.patches as mpatches
# 进行预测
data_predict = ma1.predict(1, 371)
# print(data_predict[-11:]) # 输出最后的数据

init_data = pd.read_excel(path, sheet_name='NH.Ts+dSST', header=1, index_col=0).loc[1990:2020, 'Jan':'Dec']
plt.figure(figsize=(12, 8))
init_x = init_data.values.ravel()[:-1]# 最后一个数据是空的

oral_line, = plt.plot(init_x, color="b", label="原始数据")
x = list(range(len(data_predict)))
predict_line, = plt.plot(x, data_predict, color='r', label="预测数据")

# 预测数据进行填充
plt.fill_betweenx(np.arange(0.0, 2, 0.01), 360, 370, color='yellow', alpha=0.3)
handles=[oral_line, predict_line, yellow_patch]
labels=["原始数据", "预测数据", "验证区域"]
plt.legend(handles, labels, fontsize=20, loc='upper left')

# 画竖线，方便看
axvline_li = [0, 359, 360, 370]
for i in axvline_li:
    plt.axvline(i, color='k', alpha=0.3)
plt.xticks(list(range(0, len(data_predict), 12)))
plt.show()

# ================预测误差================
predict_10 = init_x[-11:]
fact_10 = data_predict[-11:]
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
mse = mean_squared_error(predict_10, fact_10)
mae = mean_absolute_error(predict_10, fact_10)
r2s = r2_score(predict_10, fact_10)
print('MSE: ', mse, '\n', 'MAE: ', mae, '\n', 'R square: ', r2s)