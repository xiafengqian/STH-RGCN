import torch
import numpy as np

def generate_dataset(X, num_timesteps_input, num_timesteps_output):

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features = []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))

    return torch.from_numpy(np.array(features))



X = np.load("data/V_sub157.npy")
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
X6 = []
X7 = []
X8 = []
for V in X:
    X1.append(
        [V[0], V[1], V[2], V[3], V[4], V[5], V[17], V[18], V[19], V[20], V[21], V[22], V[23], V[25], V[26],
         V[30], V[31], V[32], V[33], V[34],
         V[37], V[38], V[39], V[40], V[41], V[43], V[44], V[45], V[47], V[48], V[50], V[52], V[53], V[55],
         V[56], V[59], V[60], V[61], V[62], V[63],
         V[64], V[65], V[66], V[67], V[68], V[69], V[73], V[74], V[75], V[76], V[77], V[78], V[79], V[80],
         V[85], V[86], V[87], V[88], V[89], V[90],
         V[92], V[93], V[94], V[95], V[96], V[97], V[98], V[99], V[101], V[102], V[103], V[104], V[106], V[108],
         V[119], V[120], V[121], V[122], V[123], V[124],
         V[126], V[132], V[133], V[134], V[135], V[137], V[138], V[139], V[140], V[141], V[142], V[143], V[144],
         V[145], V[146], V[147], V[148], V[149], V[150], V[152],
         V[154], V[155], V[156]])
X1 = np.array(X1)
for V in X:
    X2.append([V[11], V[12], V[13], V[14], V[15], V[16], V[24], V[27], V[28], V[29], V[35], V[49], V[70], V[71],
               V[72], V[112], V[113], V[127], V[129], V[130],
               V[131], V[136], V[151], V[153]])
X2 = np.array(X2)
for V in X:
    X3.append([V[6], V[7], V[36], V[25]])
X3 = np.array(X3)
for V in X:
    X4.append([V[42], V[54], V[91], V[100], V[107], V[109], V[110], V[114]])
X4 = np.array(X4)
for V in X:
    X5.append([V[9], V[10], V[51], V[111]])
X5 = np.array(X5)
for V in X:
    X6.append([V[8], V[57], V[81], V[84], V[125]])
X6 = np.array(X6)
for V in X:
    X7.append([V[82], V[83]])
X7 = np.array(X7)
for V in X:
    X8.append([V[46], V[105], V[115], V[116], V[117], V[118], V[128]])
X8 = np.array(X8)

X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))
X2 = np.reshape(X2, (X2.shape[0], X2.shape[1], 1))
X3 = np.reshape(X3, (X3.shape[0], X3.shape[1], 1))
X4 = np.reshape(X4, (X4.shape[0], X4.shape[1], 1))
X5 = np.reshape(X5, (X5.shape[0], X5.shape[1], 1))
X6 = np.reshape(X6, (X6.shape[0], X6.shape[1], 1))
X7 = np.reshape(X7, (X7.shape[0], X7.shape[1], 1))
X8 = np.reshape(X8, (X8.shape[0], X8.shape[1], 1))

X1 = X1.transpose((1, 2, 0))
X2 = X2.transpose((1, 2, 0))
X3 = X3.transpose((1, 2, 0))
X4 = X4.transpose((1, 2, 0))
X5 = X5.transpose((1, 2, 0))
X6 = X6.transpose((1, 2, 0))
X7 = X7.transpose((1, 2, 0))
X8 = X8.transpose((1, 2, 0))

X1 = X1.astype(np.float32)
X2 = X2.astype(np.float32)
X3 = X3.astype(np.float32)
X4 = X4.astype(np.float32)
X5 = X5.astype(np.float32)
X6 = X6.astype(np.float32)
X7 = X7.astype(np.float32)
X8 = X8.astype(np.float32)

means1 = np.mean(X1, axis=(0, 2))
means2 = np.mean(X2, axis=(0, 2))
means3 = np.mean(X3, axis=(0, 2))
means4 = np.mean(X4, axis=(0, 2))
means5 = np.mean(X5, axis=(0, 2))
means6 = np.mean(X6, axis=(0, 2))
means7 = np.mean(X7, axis=(0, 2))
means8 = np.mean(X8, axis=(0, 2))

stds1 = np.std(X1, axis=(0, 2))
stds2 = np.std(X2, axis=(0, 2))
stds3 = np.std(X3, axis=(0, 2))
stds4 = np.std(X4, axis=(0, 2))
stds5 = np.std(X5, axis=(0, 2))
stds6 = np.std(X6, axis=(0, 2))
stds7 = np.std(X7, axis=(0, 2))
stds8 = np.std(X8, axis=(0, 2))

X1 = X1 - means1.reshape(1, -1, 1)
X2 = X2 - means2.reshape(1, -1, 1)
X3 = X3 - means3.reshape(1, -1, 1)
X4 = X4 - means4.reshape(1, -1, 1)
X5 = X5 - means5.reshape(1, -1, 1)
X6 = X6 - means6.reshape(1, -1, 1)
X7 = X7 - means7.reshape(1, -1, 1)
X8 = X8 - means8.reshape(1, -1, 1)

X1 = X1 / stds1.reshape(1, -1, 1)
X2 = X2 / stds2.reshape(1, -1, 1)
X3 = X3 / stds3.reshape(1, -1, 1)
X4 = X4 / stds4.reshape(1, -1, 1)
X5 = X5 / stds5.reshape(1, -1, 1)
X6 = X6 / stds6.reshape(1, -1, 1)
X7 = X7 / stds7.reshape(1, -1, 1)
X8 = X8 / stds8.reshape(1, -1, 1)


X1 =torch.tensor(X1)
X2 =torch.tensor(X2)
X3 =torch.tensor(X3)
X4 =torch.tensor(X4)
X5 =torch.tensor(X5)
X6 =torch.tensor(X6)
X7 =torch.tensor(X7)
X8 =torch.tensor(X8)
