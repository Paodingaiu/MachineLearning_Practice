from cmath import log
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

""" practice a simple linear regression model
"""
# config logging sets
logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename="./logs/LinearRegression.log" #log日志输出的文件位置和文件名
                    ,filemode="a" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )

# parpare for data and vectorization
def preapare_for_data(data:pd.DataFrame):
    dataMean = data.mean()
    dataSdDeviation = data.std()
    out_data = (data - dataMean)/dataSdDeviation
    return out_data


# define a model


# use gradient descent to get a cost graph , find a minimize cost funcation with theta


# goal
def train(data):
    theta = np.empty((data.shape[1]+1,1),dtype=np.float32)
    print(theta)

# main app
logging.info("training starts")
logging.info('loading data...')
predata = pd.read_csv(r'./data/world-happiness-report-2017.csv',encoding='gb2312')
x1Name = "Economy..GDP.per.Capita."
x2Name = "Health..Life.Expectancy."
x3Name = "Freedom"
yName = "Happiness.Score"

logging.info(predata.info())

constantVariable = np.ones((predata.shape[0]))
data = preapare_for_data(predata.loc[:,(x1Name,x2Name,x3Name)])
data = data.insert(1,'theta',constantVariable)
y = preapare_for_data(predata.loc[:,yName])

# plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()
ax.scatter(data.loc[:,x1Name], y, linewidth=2.0,label=x1Name)
ax.scatter(data.loc[:,x2Name], y, linewidth=2.0,label=x2Name,marker='x')
ax.scatter(data.loc[:,x3Name], y, linewidth=2.0,label=x3Name,marker='*')
ax.set_title("the Source Data")
# plt.xlabel(x1Name)
plt.ylabel(yName)
plt.legend()
plt.show()

# start training
train(data)




logging.info("training ends")



