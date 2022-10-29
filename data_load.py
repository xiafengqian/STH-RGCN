# encoding utf-8
'''
@Author: william
@Description:
@time:2020/6/15 10:52
'''
from utils import generate_dataset, load_metr_la_data
import pandas as pd
import numpy as np
import torch

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data


def generate_dataset1(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features = []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))

    return torch.from_numpy(np.array(features))

def Data_load(num_timesteps_input, num_timesteps_output):
    # A, X, means, stds = load_metr_la_data()
    A, X, XX, X1, X2, X3, X4, X5, X6, X7, X8, means, stds, X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train, X1_val, X2_val, X3_val, X4_val, X5_val, X6_val, X7_val, X8_val = load_metr_la_data()

    split_line1 = int(XX.shape[2] * 0.6)
    split_line2 = int(XX.shape[2] * 0.8)

    train_original_data = XX[:, :, :split_line1]
    val_original_data = XX[:, :, split_line1:split_line2]
    test_original_data = XX[:, :, split_line2:]

    val_original_data2 = XX[:, :, split_line1:split_line1 + 52]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)

    val_input2, val_target2 = generate_dataset(val_original_data2,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)

    X1 = generate_dataset1(X1, num_timesteps_input=12, num_timesteps_output=9)
    X2 = generate_dataset1(X2, num_timesteps_input=12, num_timesteps_output=9)
    X3 = generate_dataset1(X3, num_timesteps_input=12, num_timesteps_output=9)
    X4 = generate_dataset1(X4, num_timesteps_input=12, num_timesteps_output=9)
    X5 = generate_dataset1(X5, num_timesteps_input=12, num_timesteps_output=9)
    X6 = generate_dataset1(X6, num_timesteps_input=12, num_timesteps_output=9)
    X7 = generate_dataset1(X7, num_timesteps_input=12, num_timesteps_output=9)
    X8 = generate_dataset1(X8, num_timesteps_input=12, num_timesteps_output=9)

    X1_train = generate_dataset1(X1_train, num_timesteps_input=12, num_timesteps_output=9)
    X2_train = generate_dataset1(X2_train, num_timesteps_input=12, num_timesteps_output=9)
    X3_train = generate_dataset1(X3_train, num_timesteps_input=12, num_timesteps_output=9)
    X4_train = generate_dataset1(X4_train, num_timesteps_input=12, num_timesteps_output=9)
    X5_train = generate_dataset1(X5_train, num_timesteps_input=12, num_timesteps_output=9)
    X6_train = generate_dataset1(X6_train, num_timesteps_input=12, num_timesteps_output=9)
    X7_train = generate_dataset1(X7_train, num_timesteps_input=12, num_timesteps_output=9)
    X8_train = generate_dataset1(X8_train, num_timesteps_input=12, num_timesteps_output=9)

    X1_val = generate_dataset1(X1_val, num_timesteps_input=12, num_timesteps_output=9)
    X2_val = generate_dataset1(X2_val, num_timesteps_input=12, num_timesteps_output=9)
    X3_val = generate_dataset1(X3_val, num_timesteps_input=12, num_timesteps_output=9)
    X4_val = generate_dataset1(X4_val, num_timesteps_input=12, num_timesteps_output=9)
    X5_val = generate_dataset1(X5_val, num_timesteps_input=12, num_timesteps_output=9)
    X6_val = generate_dataset1(X6_val, num_timesteps_input=12, num_timesteps_output=9)
    X7_val = generate_dataset1(X7_val, num_timesteps_input=12, num_timesteps_output=9)
    X8_val = generate_dataset1(X8_val, num_timesteps_input=12, num_timesteps_output=9)

    X1 = X1.transpose(0, 1);X2 = X2.transpose(0, 1);X3 = X3.transpose(0, 1);X4 = X4.transpose(0, 1);X5 = X5.transpose(0, 1);X6 = X6.transpose(0, 1);X7 = X7.transpose(0, 1);X8 = X8.transpose(0, 1)
    X1 = X1.transpose(1, 3);X2 = X2.transpose(1, 3);X3 = X3.transpose(1, 3);X4 = X4.transpose(1, 3);X5 = X5.transpose(1, 3);X6 = X6.transpose(1, 3);X7 = X7.transpose(1, 3);X8 = X8.transpose(1, 3)

    X1_train = X1_train.transpose(0, 1);X2_train = X2_train.transpose(0, 1);X3_train = X3_train.transpose(0, 1);X4_train = X4_train.transpose(0, 1);X5_train = X5_train.transpose(0, 1);X6_train = X6_train.transpose(0, 1);X7_train = X7_train.transpose(0, 1);X8_train = X8_train.transpose(0, 1)
    X1_train = X1_train.transpose(1, 3);X2_train = X2_train.transpose(1, 3);X3_train = X3_train.transpose(1, 3);X4_train = X4_train.transpose(1, 3);X5_train = X5_train.transpose(1, 3);X6_train = X6_train.transpose(1, 3);X7_train = X7_train.transpose(1, 3);X8_train = X8_train.transpose(1, 3)

    X1_val = X1_val.transpose(0, 1);X2_val = X2_val.transpose(0, 1);X3_val = X3_val.transpose(0, 1);X4_val = X4_val.transpose(0, 1);X5_val = X5_val.transpose(0, 1);X6_val = X6_val.transpose(0, 1);X7_val = X7_val.transpose(0, 1);X8_val = X8_val.transpose(0, 1)
    X1_val = X1_val.transpose(1, 3);X2_val = X2_val.transpose(1, 3);X3_val = X3_val.transpose(1, 3);X4_val = X4_val.transpose(1, 3);X5_val = X5_val.transpose(1, 3);X6_val = X6_val.transpose(1, 3);X7_val = X7_val.transpose(1, 3);X8_val = X8_val.transpose(1, 3)

    # print(X1_train.shape)
    # print(X1_val.shape)
    return A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target, val_input2, val_target2, X1, X2, X3, X4, X5, X6, X7, X8, X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train, X1_val, X2_val, X3_val, X4_val, X5_val, X6_val, X7_val, X8_val

