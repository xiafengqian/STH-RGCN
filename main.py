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

from STHRGCN import STGCN
from utils import get_normalized_adj, RMSE, MAE, MAPE
from data_load import Data_load

num_timesteps_input = 12
num_timesteps_output = 9

epochs = 100
batch_size = 64

# 设置使用的GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='STGCN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
device=torch.device("cuda:0")

def train_epoch(training_input, training_target, X_dict, X1, X2, X3, X4, X5, X6, X7, X8, batch_size):
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
    for i in range(0, training_input.shape[0] - 18, batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]

        x1= X1[:, indices, :, :];x2= X2[:, indices, :, :];x3= X3[:, indices, :, :];x4= X4[:, indices, :, :]
        x5= X5[:, indices, :, :];x6= X6[:, indices, :, :];x7= X7[:, indices, :, :];x8= X8[:, indices, :, :]

        X_dict = {'type1': x1, 'type2': x2, 'type3': x3, 'type4': x4, 'type5': x5, 'type6': x6, 'type7': x7, 'type8': x8}
        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
            # X_dict = X_dict.cuda()
        out = net(A_wave, X_dict)
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
        # validation_losses.append(np.asscalar(val_loss.detach().numpy()))
        validation_losses.append(np.ndarray.item(val_loss.detach().numpy()))
        out_unnormalized = out_index.detach().cpu().numpy() * stds[0] + means[0]
        target_unnormalized = val_target_index.detach().cpu().numpy() * stds[0] + means[0]

        if (epoch % 5 == 0) & (epoch != 0):
            np.savetxt("./results_STGCN/DeepUVI_s/pred_result_" + str(epoch) + ".csv", out_index.cpu(), delimiter=',')
            np.savetxt("./results_STGCN/DeepUVI_s/true_result_" + str(epoch) + ".csv", val_target_index.cpu(), delimiter=',')

        mae = MAE(out_unnormalized, target_unnormalized)
        validation_MAE.append(mae)

        rmse = RMSE(out_unnormalized, target_unnormalized)
        validation_RMSE.append(rmse)

        mape = MAPE(out_unnormalized, target_unnormalized)
        validation_MAPE.append(mape)

    return validation_losses, validation_MAE, validation_RMSE, validation_MAPE


def cal_acc():
    pass


if __name__ == '__main__':

    torch.manual_seed(7)

    g, A, means, stds, training_input, training_target, val_input, val_target, test_input, test_target, val_input2, val_target2, X_dict, X_train_dict, X_val_dict, X, X_train, X_val, X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train\
        = Data_load(num_timesteps_input, num_timesteps_output)
    torch.cuda.empty_cache()    # free cuda memory

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    if torch.cuda.is_available():
        A_wave = A_wave.cuda()
        training_input = training_input.cuda()
        training_target = training_target.cuda()
    net = STGCN(A_wave.shape[0],
                training_input.shape[3],
                num_timesteps_input,
                num_timesteps_output,
                hg=g
                )

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
        loss = train_epoch(training_input, training_target, X_train_dict, X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train,
                           batch_size=batch_size)
        training_losses.append(loss)
        torch.cuda.empty_cache()  # free cuda memory
        # Run validation
        #torch.no_grad()两个作用：新增的tensor没有梯度，使带梯度的tensor能够进行原地运算。
        with torch.no_grad():
            net.eval()
            if torch.cuda.is_available():
                val_input = val_input.cuda()
                val_target = val_target.cuda()

            out = net(A_wave, X_val_dict)

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
        plt.subplot(2, 1, 1)
        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(validation_RMSE, label="validation_RMSE")
        plt.legend()

        plt.show()
        # checkpoint_path = "checkpoints/"
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path)
        # with open("checkpoints/losses.pk", "wb") as fd:
        #     pk.dump((training_losses, validation_losses, validation_MAE), fd)
