# encoding utf-8
'''
@Author: william
@Description: CSV转NPZ
@time:2020/6/16 19:51
'''
import pandas as pd
import numpy as np

data_path1 = "w_157.csv"
data_path2 = "cll.csv"
data1 = pd.read_csv(data_path1, header=None)
data2 = pd.read_csv(data_path2, header=None)
print(data1.shape)
print(data1.head())

# data.drop(columns=387, inplace=True)

print(data2.shape)
print(data2.head())

np.save("../data/W_sub157.npy", data1)
np.save("../data/V_sub157.npy", data2)