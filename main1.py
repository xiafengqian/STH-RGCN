import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import dgl
from dgl.data import DGLDataset
import pandas as pd

from stgcn import STGCN
from utils import get_normalized_adj, RMSE, MAE, MAPE
from data_load import Data_load

num_timesteps_input = 12
num_timesteps_output = 9

epochs = 50
batch_size = 64

# 设置使用的GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    loss_mean = 0.0
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
        # print('loss: ' + str(loss_mean))
    return loss_mean


def Cal_eval_index(out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE):
    for item in arr:
        out_index = out[:, :, item - 1]
        val_target_index = val_target[:, :, item - 1]

        val_loss = loss_criterion(out_index, val_target_index).to(device="cpu")
        validation_losses.append(np.asscalar(val_loss.detach().numpy()))

        out_unnormalized = out_index.detach().cpu().numpy() * stds[0] + means[0]
        target_unnormalized = val_target_index.detach().cpu().numpy() * stds[0] + means[0]

        mae = MAE(out_unnormalized, target_unnormalized)
        validation_MAE.append(mae)

        rmse = RMSE(out_unnormalized, target_unnormalized)
        validation_RMSE.append(rmse)

        mape = MAPE(out_unnormalized, target_unnormalized)
        validation_MAPE.append(mape)

    return validation_losses, validation_MAE, validation_RMSE, validation_MAPE


def cal_acc():
    pass

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data

if __name__ == '__main__':

    i=0;j=0;k=0;l=0

    i = 0;
    j = 0


    def load_csv(path):  # 读csv文件，转化为数组
        data_read = pd.read_csv(path)  # 返回dataframe
        list = data_read.values.tolist()  # 变成list
        data = np.array(list)  # 数组
        # print(data.shape)
        # print(data)
        return data


    C = [[0] * 10] * 1000
    D = [[0] * 10] * 500
    B = load_csv('cll.csv')
    # print(C)
    for i in range(1000):
        for j in range(10):
            C[i][j] = B[i][j]
    for i in range(500):
        for j in range(10):
            D[i][j] = B[i][j]
    C = np.array(C)
    D = np.array(D)
    # print(B.shape)
    # print(C.shape)
    # print(D.shape)

    n_users = 100
    n_items = 57
    n_follows = 300
    n_clicks = 500
    n_dislikes = 50
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    g = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})
    print(g)

    torch.manual_seed(7)
    A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target \
        = Data_load(num_timesteps_input, num_timesteps_output)
    torch.cuda.empty_cache()    # free cuda memory

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    if torch.cuda.is_available():
        A_wave = A_wave.cuda()

    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output,
                hg=g)
    if torch.cuda.is_available():
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_MAE = []
    validation_RMSE = []
    validation_MAPE = []
    arr = [3, 6, 9]     # 对应15分钟，30分钟，45分钟
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        training_losses.append(loss)
        torch.cuda.empty_cache()  # free cuda memory
        # Run validation
        with torch.no_grad():
            net.eval()
            if torch.cuda.is_available():
                val_input = val_input.cuda()
                val_target = val_target.cuda()
            out = net(A_wave, val_input)
            print(out.shape)

            validation_losses, validation_MAE, validation_RMSE, validation_MAPE = \
                Cal_eval_index(out, val_target, arr, validation_losses, validation_MAE, validation_RMSE, validation_MAPE)

            out = None
            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        print(epoch+1)
        print("Training loss: {}".format(training_losses[-1]))
        for i in range(len(arr)):
            print("time slice:{}, Validation loss:{}, MAE:{}, MAPE:{}, RMSE:{}"
                  .format(arr[i], validation_losses[-(len(arr) - i)], validation_MAE[-(len(arr) - i)],
                          validation_MAPE[-(len(arr) - i)], validation_RMSE[-(len(arr) - i)],))

        # plt.plot(training_losses, label="training loss")
        # plt.plot(validation_losses, label="validation loss")
        # plt.legend()
        # plt.show()

        # checkpoint_path = "checkpoints/"
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path)
        # with open("checkpoints/losses.pk", "wb") as fd:
        #     pk.dump((training_losses, validation_losses, validation_MAE), fd)
