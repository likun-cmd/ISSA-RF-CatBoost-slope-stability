
from sklearn import preprocessing
import pandas as pd
import numpy as np
# -*- coding: utf-8 -*-

data = pd.read_excel("raw_data.xlsx")#输入训练集
x=data.iloc[:,0:6]
print(x)
min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
print(np.around(x_minmax,decimals=3))


testData=np.around(x_minmax,decimals=3)

def pd_toExcel(data, fileName):  # pandas库储存数据到excel
    a = []
    b = []
    c = []
    d= []
    e = []
    f = []
    # g = []
    for i in range(len(data)):
        a.append(data[i][0])
        b.append(data[i][1])
        c.append(data[i][2])
        d.append(data[i][3])
        e.append(data[i][4])
        f.append(data[i][5])
        # g.append(data[i][6])

    dfData = {  # 用字典设置DataFrame所需数据
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f,
        # 'g': g
    }
    df = pd.DataFrame(dfData)  # 创建DataFrame
    df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）



fileName = 'data_set.xlsx'
pd_toExcel(testData, fileName)