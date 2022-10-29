import numpy as np
import pandas as pd
import os


def fill_nan(data):
    if data.isnull().values.any():
        data.fillna(method='ffill', inplace=True)
    if data.isnull().values.any():
        data.fillna(method='bfill', inplace=True)


# path_name = r"E:\C\大四\毕业设计\data\2021-01-28-晚上-数据集157个点\数据集157个点"
# occ_path = path_name + "/占用率"
# speed_path = path_name + "/车速"

path_name = r"E:\C\大四\毕业设计\data\数据集387个点"
occ_path = path_name + "/occ"
speed_path = path_name + "/speed"

flow_file_name = os.listdir(occ_path)

v_occ_path = occ_path + "/" + flow_file_name[0]
print(v_occ_path)

speed_file_name = os.listdir(speed_path)
v_speed_path = speed_path + "/" + speed_file_name[0]
print(v_speed_path)


# v_occ_path = r'E:\C\大四\毕业设计\data\80\V_occ_20-11-01_20-12-15_80.xlsx'
# v_speed_path = r'E:\C\大四\毕业设计\data\80\V_speed_20-11-01_20-12-15_80.xlsx'

v_flow = pd.read_excel(v_occ_path, header=None)    # 占用率数据
print(v_flow.shape)

v_speed = pd.read_excel(v_speed_path, header=None)  # 速度数据
print(v_speed.shape)

fill_nan(v_flow)
fill_nan(v_speed)

v_flow = v_flow.values
v_speed = v_speed.values

print(type(v_flow))
c = np.stack((v_speed, v_flow), axis=2)
print(c.shape)

np.save("../data/V_matrix_387.npy", c)

