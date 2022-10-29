import os
import zipfile
import numpy as np
import pandas as pd
import torch
#from p_test import get_p_score

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data

def load_metr_la_data():
    # if (not os.path.isfile("data/adj_mat.npy")
    #         or not os.path.isfile("data/node_values.npy")):
    #     with zipfile.ZipFile("data/METR-LA.zip", 'r') as zip_ref:
    #         zip_ref.extractall("data/")

    # get_p_score(1000, 0.9, 0.7)   # 记得改路径！！！！
    A = np.load("data/W_sub157.npy")
    # A=load_csv('w_157.csv')
    # A = np.load("data/ptest_W_matrix_80.npy")
    X = np.load("data/V_sub157.npy")
    X1 = [];X2 = [];X3 = [];X4 = [];X5 = [];X6 = [];X7 = [];X8 = [];XX=[]
    for V in X:
        X1.append([V[0],V[1],V[2],V[3],V[4],V[5],V[17],V[18],V[19],V[20],V[21],V[22],V[23],V[25],V[26],V[30],V[31],V[32],V[33],V[34],
                   V[37],V[38],V[39],V[40],V[41],V[43],V[44],V[45],V[47],V[48],V[50],V[52],V[53],V[55],V[56],V[59],V[60],V[61],V[62],V[63],
                   V[64],V[65],V[66],V[67],V[68],V[69],V[73],V[74],V[75],V[76],V[77],V[78],V[79],V[80],V[85],V[86],V[87],V[88],V[89],V[90],
                   V[92],V[93],V[94],V[95],V[96],V[97],V[98],V[99],V[101],V[102],V[103],V[104],V[106],V[108],V[119],V[120],V[121],V[122],V[123],V[124],
                   V[126],V[132],V[133],V[134],V[135],V[137],V[138],V[139],V[140],V[141],V[142],V[143],V[144],V[145],V[146],V[147],V[148],V[149],V[150],V[152],
                   V[154],V[155],V[156]])
    X1=np.array(X1)
    # print(X1.shape)
    for V in X:
        X2.append([V[11],V[12],V[13],V[14],V[15],V[16],V[24],V[27],V[28],V[29],V[35],V[49],V[70],V[71],V[72],V[112],V[113],V[127],V[129],V[130],
                   V[131],V[136],V[151],V[153]])
    X2=np.array(X2)
    # print(X2.shape)
    for V in X:
        X3.append([V[6],V[7],V[36],V[25]])
    X3 = np.array(X3)
    # print(X3.shape)
    for V in X:
        X4.append([V[42],V[54],V[91],V[100],V[107],V[109],V[110],V[114]])
    X4 = np.array(X4)
    # print(X4.shape)
    for V in X:
        X5.append([V[9],V[10],V[51],V[111]])
    X5 = np.array(X5)
    # print(X5.shape)
    for V in X:
        X6.append([V[8],V[57],V[81],V[84],V[125]])
    X6 = np.array(X6)
    # print(X6.shape)
    for V in X:
        X7.append([V[82],V[83]])
    X7 = np.array(X7)
    # print(X7.shape)
    for V in X:
        X8.append([V[46],V[105],V[115],V[116],V[117],V[118],V[128]])
    X8 = np.array(X8)

    for V in X:
        XX.append([V[0],V[1],V[2],V[3],V[4],V[5],V[17],V[18],V[19],V[20],V[21],V[22],V[23],V[25],V[26],V[30],V[31],V[32],V[33],V[34],
                   V[37],V[38],V[39],V[40],V[41],V[43],V[44],V[45],V[47],V[48],V[50],V[52],V[53],V[55],V[56],V[59],V[60],V[61],V[62],V[63],
                   V[64],V[65],V[66],V[67],V[68],V[69],V[73],V[74],V[75],V[76],V[77],V[78],V[79],V[80],V[85],V[86],V[87],V[88],V[89],V[90],
                   V[92],V[93],V[94],V[95],V[96],V[97],V[98],V[99],V[101],V[102],V[103],V[104],V[106],V[108],V[119],V[120],V[121],V[122],V[123],V[124],
                   V[126],V[132],V[133],V[134],V[135],V[137],V[138],V[139],V[140],V[141],V[142],V[143],V[144],V[145],V[146],V[147],V[148],V[149],V[150],V[152],
                   V[154],V[155],V[156],V[11],V[12],V[13],V[14],V[15],V[16],V[24],V[27],V[28],V[29],V[35],V[49],V[70],V[71],V[72],V[112],V[113],V[127],V[129],V[130],
                   V[131],V[136],V[151],V[153],V[6],V[7],V[36],V[25],V[42],V[54],V[91],V[100],V[107],V[109],V[110],V[114],V[9],V[10],V[51],V[111],
                   V[8],V[57],V[81],V[84],V[125],V[82],V[83],V[46],V[105],V[115],V[116],V[117],V[118],V[128]])
    XX = np.array(XX)
    # print(XX.shape)
    # X=load_csv('cll.csv')

    # X = np.reshape(X, (X.shape[0], X.shape[1], 1)).transpose((1, 2, 0))
    split_line1 = int(XX.shape[0] * 0.6)
    split_line2 = int(XX.shape[0] * 0.8)
    train_original_data = XX[ :split_line1, :]
    val_original_data = XX[split_line1:split_line2, :]

    X1_train = [];X2_train = [];X3_train = [];X4_train = [];X5_train = [];X6_train = [];X7_train = [];X8_train = [];
    X1_val = [];X2_val = [];X3_val = [];X4_val = [];X5_val = [];X6_val = [];X7_val = [];X8_val = [];

    for V in train_original_data:
        X1_train.append(
            [V[0], V[1], V[2], V[3], V[4], V[5], V[17], V[18], V[19], V[20], V[21], V[22], V[23], V[25], V[26], V[30],
             V[31], V[32], V[33], V[34],
             V[37], V[38], V[39], V[40], V[41], V[43], V[44], V[45], V[47], V[48], V[50], V[52], V[53], V[55], V[56],
             V[59], V[60], V[61], V[62], V[63],
             V[64], V[65], V[66], V[67], V[68], V[69], V[73], V[74], V[75], V[76], V[77], V[78], V[79], V[80], V[85],
             V[86], V[87], V[88], V[89], V[90],
             V[92], V[93], V[94], V[95], V[96], V[97], V[98], V[99], V[101], V[102], V[103], V[104], V[106], V[108],
             V[119], V[120], V[121], V[122], V[123], V[124],
             V[126], V[132], V[133], V[134], V[135], V[137], V[138], V[139], V[140], V[141], V[142], V[143], V[144],
             V[145], V[146], V[147], V[148], V[149], V[150], V[152],
             V[154], V[155], V[156]])
    X1_train = np.array(X1_train)
    # print(X1_train.shape)
    for V in train_original_data:
        X2_train.append(
            [V[11], V[12], V[13], V[14], V[15], V[16], V[24], V[27], V[28], V[29], V[35], V[49], V[70], V[71], V[72],
             V[112], V[113], V[127], V[129], V[130],
             V[131], V[136], V[151], V[153]])
    X2_train = np.array(X2_train)
    # print(X2_train.shape)
    for V in train_original_data:
        X3_train.append([V[6], V[7], V[36], V[25]])
    X3_train = np.array(X3_train)
    # print(X3_train.shape)
    for V in train_original_data:
        X4_train.append([V[42], V[54], V[91], V[100], V[107], V[109], V[110], V[114]])
    X4_train = np.array(X4_train)
    # print(X4_train.shape)
    for V in train_original_data:
        X5_train.append([V[9], V[10], V[51], V[111]])
    X5_train = np.array(X5_train)
    # print(X5_train.shape)
    for V in train_original_data:
        X6_train.append([V[8], V[57], V[81], V[84], V[125]])
    X6_train = np.array(X6_train)
    # print(X6_train.shape)
    for V in train_original_data:
        X7_train.append([V[82], V[83]])
    X7_train = np.array(X7_train)
    # print(X7_train.shape)
    for V in train_original_data:
        X8_train.append([V[46], V[105], V[115], V[116], V[117], V[118], V[128]])
    X8_train = np.array(X8_train)
    # print(X8_train.shape)

    X1_train = np.reshape(X1_train, (X1_train.shape[0], X1_train.shape[1], 1))
    X2_train = np.reshape(X2_train, (X2_train.shape[0], X2_train.shape[1], 1))
    X3_train = np.reshape(X3_train, (X3_train.shape[0], X3_train.shape[1], 1))
    X4_train = np.reshape(X4_train, (X4_train.shape[0], X4_train.shape[1], 1))
    X5_train = np.reshape(X5_train, (X5_train.shape[0], X5_train.shape[1], 1))
    X6_train = np.reshape(X6_train, (X6_train.shape[0], X6_train.shape[1], 1))
    X7_train = np.reshape(X7_train, (X7_train.shape[0], X7_train.shape[1], 1))
    X8_train = np.reshape(X8_train, (X8_train.shape[0], X8_train.shape[1], 1))

    X1_train = X1_train.transpose((1, 2, 0))
    X2_train = X2_train.transpose((1, 2, 0))
    X3_train = X3_train.transpose((1, 2, 0))
    X4_train = X4_train.transpose((1, 2, 0))
    X5_train = X5_train.transpose((1, 2, 0))
    X6_train = X6_train.transpose((1, 2, 0))
    X7_train = X7_train.transpose((1, 2, 0))
    X8_train = X8_train.transpose((1, 2, 0))

    X1_train = X1_train.astype(np.float32)
    X2_train = X2_train.astype(np.float32)
    X3_train = X3_train.astype(np.float32)
    X4_train = X4_train.astype(np.float32)
    X5_train = X5_train.astype(np.float32)
    X6_train = X6_train.astype(np.float32)
    X7_train = X7_train.astype(np.float32)
    X8_train = X8_train.astype(np.float32)

    means1 = np.mean(X1_train, axis=(0, 2))
    means2 = np.mean(X2_train, axis=(0, 2))
    means3 = np.mean(X3_train, axis=(0, 2))
    means4 = np.mean(X4_train, axis=(0, 2))
    means5 = np.mean(X5_train, axis=(0, 2))
    means6 = np.mean(X6_train, axis=(0, 2))
    means7 = np.mean(X7_train, axis=(0, 2))
    means8 = np.mean(X8_train, axis=(0, 2))

    stds1 = np.std(X1_train, axis=(0, 2))
    stds2 = np.std(X2_train, axis=(0, 2))
    stds3 = np.std(X3_train, axis=(0, 2))
    stds4 = np.std(X4_train, axis=(0, 2))
    stds5 = np.std(X5_train, axis=(0, 2))
    stds6 = np.std(X6_train, axis=(0, 2))
    stds7 = np.std(X7_train, axis=(0, 2))
    stds8 = np.std(X8_train, axis=(0, 2))

    X1_train = X1_train - means1.reshape(1, -1, 1)
    X2_train = X2_train - means2.reshape(1, -1, 1)
    X3_train = X3_train - means3.reshape(1, -1, 1)
    X4_train = X4_train - means4.reshape(1, -1, 1)
    X5_train = X5_train - means5.reshape(1, -1, 1)
    X6_train = X6_train - means6.reshape(1, -1, 1)
    X7_train = X7_train - means7.reshape(1, -1, 1)
    X8_train = X8_train - means8.reshape(1, -1, 1)

    X1_train = X1_train / stds1.reshape(1, -1, 1)
    X2_train = X2_train / stds2.reshape(1, -1, 1)
    X3_train = X3_train / stds3.reshape(1, -1, 1)
    X4_train = X4_train / stds4.reshape(1, -1, 1)
    X5_train = X5_train / stds5.reshape(1, -1, 1)
    X6_train = X6_train / stds6.reshape(1, -1, 1)
    X7_train = X7_train / stds7.reshape(1, -1, 1)
    X8_train = X8_train / stds8.reshape(1, -1, 1)

    for V in val_original_data:
        X1_val.append(
            [V[0], V[1], V[2], V[3], V[4], V[5], V[17], V[18], V[19], V[20], V[21], V[22], V[23], V[25], V[26], V[30],
             V[31], V[32], V[33], V[34],
             V[37], V[38], V[39], V[40], V[41], V[43], V[44], V[45], V[47], V[48], V[50], V[52], V[53], V[55], V[56],
             V[59], V[60], V[61], V[62], V[63],
             V[64], V[65], V[66], V[67], V[68], V[69], V[73], V[74], V[75], V[76], V[77], V[78], V[79], V[80], V[85],
             V[86], V[87], V[88], V[89], V[90],
             V[92], V[93], V[94], V[95], V[96], V[97], V[98], V[99], V[101], V[102], V[103], V[104], V[106], V[108],
             V[119], V[120], V[121], V[122], V[123], V[124],
             V[126], V[132], V[133], V[134], V[135], V[137], V[138], V[139], V[140], V[141], V[142], V[143], V[144],
             V[145], V[146], V[147], V[148], V[149], V[150], V[152],
             V[154], V[155], V[156]])
    X1_val = np.array(X1_val)
    # print(X1_val.shape)
    for V in val_original_data:
        X2_val.append(
            [V[11], V[12], V[13], V[14], V[15], V[16], V[24], V[27], V[28], V[29], V[35], V[49], V[70], V[71], V[72],
             V[112], V[113], V[127], V[129], V[130],
             V[131], V[136], V[151], V[153]])
    X2_val = np.array(X2_val)
    # print(X2_val.shape)
    for V in val_original_data:
        X3_val.append([V[6], V[7], V[36], V[25]])
    X3_val = np.array(X3_val)
    # print(X3_val.shape)
    for V in val_original_data:
        X4_val.append([V[42], V[54], V[91], V[100], V[107], V[109], V[110], V[114]])
    X4_val = np.array(X4_val)
    # print(X4_val.shape)
    for V in val_original_data:
        X5_val.append([V[9], V[10], V[51], V[111]])
    X5_val = np.array(X5_val)
    # print(X5_val.shape)
    for V in val_original_data:
        X6_val.append([V[8], V[57], V[81], V[84], V[125]])
    X6_val = np.array(X6_val)
    # print(X6_val.shape)
    for V in val_original_data:
        X7_val.append([V[82], V[83]])
    X7_val = np.array(X7_val)
    # print(X7_val.shape)
    for V in val_original_data:
        X8_val.append([V[46], V[105], V[115], V[116], V[117], V[118], V[128]])
    X8_val = np.array(X8_val)
    # print(X8_val.shape)

    X1_val = np.reshape(X1_val, (X1_val.shape[0], X1_val.shape[1], 1))
    X2_val = np.reshape(X2_val, (X2_val.shape[0], X2_val.shape[1], 1))
    X3_val = np.reshape(X3_val, (X3_val.shape[0], X3_val.shape[1], 1))
    X4_val = np.reshape(X4_val, (X4_val.shape[0], X4_val.shape[1], 1))
    X5_val = np.reshape(X5_val, (X5_val.shape[0], X5_val.shape[1], 1))
    X6_val = np.reshape(X6_val, (X6_val.shape[0], X6_val.shape[1], 1))
    X7_val = np.reshape(X7_val, (X7_val.shape[0], X7_val.shape[1], 1))
    X8_val = np.reshape(X8_val, (X8_val.shape[0], X8_val.shape[1], 1))

    X1_val = X1_val.transpose((1, 2, 0))
    X2_val = X2_val.transpose((1, 2, 0))
    X3_val = X3_val.transpose((1, 2, 0))
    X4_val = X4_val.transpose((1, 2, 0))
    X5_val = X5_val.transpose((1, 2, 0))
    X6_val = X6_val.transpose((1, 2, 0))
    X7_val = X7_val.transpose((1, 2, 0))
    X8_val = X8_val.transpose((1, 2, 0))

    X1_val = X1_val.astype(np.float32)
    X2_val = X2_val.astype(np.float32)
    X3_val = X3_val.astype(np.float32)
    X4_val = X4_val.astype(np.float32)
    X5_val = X5_val.astype(np.float32)
    X6_val = X6_val.astype(np.float32)
    X7_val = X7_val.astype(np.float32)
    X8_val = X8_val.astype(np.float32)

    means1 = np.mean(X1_val, axis=(0, 2))
    means2 = np.mean(X2_val, axis=(0, 2))
    means3 = np.mean(X3_val, axis=(0, 2))
    means4 = np.mean(X4_val, axis=(0, 2))
    means5 = np.mean(X5_val, axis=(0, 2))
    means6 = np.mean(X6_val, axis=(0, 2))
    means7 = np.mean(X7_val, axis=(0, 2))
    means8 = np.mean(X8_val, axis=(0, 2))

    stds1 = np.std(X1_val, axis=(0, 2))
    stds2 = np.std(X2_val, axis=(0, 2))
    stds3 = np.std(X3_val, axis=(0, 2))
    stds4 = np.std(X4_val, axis=(0, 2))
    stds5 = np.std(X5_val, axis=(0, 2))
    stds6 = np.std(X6_val, axis=(0, 2))
    stds7 = np.std(X7_val, axis=(0, 2))
    stds8 = np.std(X8_val, axis=(0, 2))

    X1_val = X1_val - means1.reshape(1, -1, 1)
    X2_val = X2_val - means2.reshape(1, -1, 1)
    X3_val = X3_val - means3.reshape(1, -1, 1)
    X4_val = X4_val - means4.reshape(1, -1, 1)
    X5_val = X5_val - means5.reshape(1, -1, 1)
    X6_val = X6_val - means6.reshape(1, -1, 1)
    X7_val = X7_val - means7.reshape(1, -1, 1)
    X8_val = X8_val - means8.reshape(1, -1, 1)

    X1_val = X1_val / stds1.reshape(1, -1, 1)
    X2_val = X2_val / stds2.reshape(1, -1, 1)
    X3_val = X3_val / stds3.reshape(1, -1, 1)
    X4_val = X4_val / stds4.reshape(1, -1, 1)
    X5_val = X5_val / stds5.reshape(1, -1, 1)
    X6_val = X6_val / stds6.reshape(1, -1, 1)
    X7_val = X7_val / stds7.reshape(1, -1, 1)
    X8_val = X8_val / stds8.reshape(1, -1, 1)


    # X = X[:, :, 1]      # 特征维度为1，只保留一个特征
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    XX = np.reshape(XX, (XX.shape[0], XX.shape[1], 1))
    X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))
    X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], 1))
    X3 = np.reshape(X3, (X3.shape[0], X3.shape[1], 1))
    X4 = np.reshape(X4, (X4.shape[0], X4.shape[1], 1))
    X5 = np.reshape(X5, (X5.shape[0], X5.shape[1], 1))
    X6 = np.reshape(X6, (X6.shape[0], X6.shape[1], 1))
    X7 = np.reshape(X7, (X7.shape[0], X7.shape[1], 1))
    X8 = np.reshape(X8, (X8.shape[0], X8.shape[1], 1))

    X = X.transpose((1, 2, 0))
    XX = XX.transpose((1, 2, 0))
    X1 = X1.transpose((1, 2, 0))
    X2 = X2.transpose((1, 2, 0))
    X3 = X3.transpose((1, 2, 0))
    X4 = X4.transpose((1, 2, 0))
    X5 = X5.transpose((1, 2, 0))
    X6 = X6.transpose((1, 2, 0))
    X7 = X7.transpose((1, 2, 0))
    X8 = X8.transpose((1, 2, 0))

    A = A.astype(np.float32)
    X = X.astype(np.float32)
    XX = XX.astype(np.float32)
    X1 = X1.astype(np.float32)
    X2 = X2.astype(np.float32)
    X3 = X3.astype(np.float32)
    X4 = X4.astype(np.float32)
    X5 = X5.astype(np.float32)
    X6 = X6.astype(np.float32)
    X7 = X7.astype(np.float32)
    X8 = X8.astype(np.float32)


    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    meansXX = np.mean(XX, axis=(0, 2))
    means1 = np.mean(X1, axis=(0, 2))
    means2 = np.mean(X2, axis=(0, 2))
    means3 = np.mean(X3, axis=(0, 2))
    means4 = np.mean(X4, axis=(0, 2))
    means5 = np.mean(X5, axis=(0, 2))
    means6 = np.mean(X6, axis=(0, 2))
    means7 = np.mean(X7, axis=(0, 2))
    means8 = np.mean(X8, axis=(0, 2))

    stds = np.std(X, axis=(0, 2))
    stdsXX = np.std(XX, axis=(0, 2))
    stds1 = np.std(X1, axis=(0, 2))
    stds2 = np.std(X2, axis=(0, 2))
    stds3 = np.std(X3, axis=(0, 2))
    stds4 = np.std(X4, axis=(0, 2))
    stds5 = np.std(X5, axis=(0, 2))
    stds6 = np.std(X6, axis=(0, 2))
    stds7 = np.std(X7, axis=(0, 2))
    stds8 = np.std(X8, axis=(0, 2))

    X = X - means.reshape(1, -1, 1)
    XX = XX - meansXX.reshape(1, -1, 1)
    X1 = X1 - means1.reshape(1, -1, 1)
    X2 = X2 - means2.reshape(1, -1, 1)
    X3 = X3 - means3.reshape(1, -1, 1)
    X4 = X4 - means4.reshape(1, -1, 1)
    X5 = X5 - means5.reshape(1, -1, 1)
    X6 = X6 - means6.reshape(1, -1, 1)
    X7 = X7 - means7.reshape(1, -1, 1)
    X8 = X8 - means8.reshape(1, -1, 1)

    X = X / stds.reshape(1, -1, 1)
    XX = XX / stdsXX.reshape(1, -1, 1)
    X1 = X1 / stds1.reshape(1, -1, 1)
    X2 = X2 / stds2.reshape(1, -1, 1)
    X3 = X3 / stds3.reshape(1, -1, 1)
    X4 = X4 / stds4.reshape(1, -1, 1)
    X5 = X5 / stds5.reshape(1, -1, 1)
    X6 = X6 / stds6.reshape(1, -1, 1)
    X7 = X7 / stds7.reshape(1, -1, 1)
    X8 = X8 / stds8.reshape(1, -1, 1)


    # flow = X[:, :, 1]
    # flow = flow.astype(np.float32)
    # means = np.mean(flow)
    #
    # flow = flow - means
    # stds = np.std(flow)
    # flow = flow / stds
    # X = flow

    return A, X, XX, X1, X2, X3, X4, X5, X6, X7, X8, means, stds, X1_train, X2_train, X3_train, X4_train, X5_train, X6_train, X7_train, X8_train, X1_val, X2_val, X3_val, X4_val, X5_val, X6_val, X7_val, X8_val


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


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


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    print(np.array(v_).shape)
    return np.mean(np.abs((v_ - v) / (v + 1e-5)))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))
