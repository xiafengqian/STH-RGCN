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

def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]
    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data
num_timesteps_input = 12
num_timesteps_output = 9


X = np.load("data/V_sub157.npy")

X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X = X.transpose((1, 2, 0))

X = X.astype(np.float32)

# Normalization using Z-score method
means = np.mean(X, axis=(0, 2))

stds = np.std(X, axis=(0, 2))

X = X - means.reshape(1, -1, 1)

# stds = np.std(X, axis=(0, 2))
X = X / stds.reshape(1, -1, 1)

split_line1 = int(X.shape[2] * 0.6)
split_line2 = int(X.shape[2] * 0.8)

train_original_data = X[:, :, :split_line1]
val_original_data = X[:, :, split_line1:split_line2]
test_original_data = X[:, :, split_line2:]

training_input, training_target = generate_dataset(train_original_data,
                                                    num_timesteps_input=num_timesteps_input,
                                                    num_timesteps_output=num_timesteps_output)
val_input, val_target = generate_dataset(val_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)
test_input, test_target = generate_dataset(test_original_data,
                                            num_timesteps_input=num_timesteps_input,
                                            num_timesteps_output=num_timesteps_output)


print(training_target.shape)
print(val_input.shape)

