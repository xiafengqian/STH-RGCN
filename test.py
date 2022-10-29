import numpy as np
import pandas as pd
import os
import pickle


# # data = np.load("data/node_values.npy")
# data = np.load("data/V_matrix_387.npy")
# print(data.shape)
#
# speed = data[:, :, 0]
# print(speed.shape)
# print(speed.min(), speed.max())
#
# flow = data[:, :, 1]
# print(flow.shape)
# print(flow.min(), flow.max())


X = np.load("data/V_matrix_387.npy")
print(X.shape)
# X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))

X = X[:, :, 1]
print(X.shape)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X = X.transpose((1, 2, 0))
print(X.shape)
