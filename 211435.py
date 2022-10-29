import numpy as np
import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super(RGCN, self).__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph,inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data

i=0;j=0;k=0;l=0
A=load_csv('w_157.csv')
B=load_csv('cll.csv')

for i in range(157):
    for j in range(i,157):
        if A[i][j]==1:
            A[j][i]=1
for i in range(157):
    for j in range(i):
        if A[i][j]==1:
            A[j][i]=1

#将节点分为八种，分别为type1:上部密集、type2：中部密集、type3：下部密集、type4：中间低两边高、
# type5：左高、type6：右高、type7：突出一部分、type8：整体都有
type=[None]*157
type1=[0,1,2,3,4,5,17,18,19,20,21,22,23,25,26,30,31,32,33,34,37,38,39,40,41,43,44,45,47,48,50,52,53,55,56,59,60,61,62,63,64,65,66,67,68,69,73,74,75,76,77,78,79,80,85,86,87,88,89,90,92,93,94,95,96,97,98,99,101,102,103,104,106,108,119,120,121,122,123,124,126,132,133,134,135,137,138,139,140,141,142,143,144,145,146,147,148,149,150,152,154,155,156]
type2=[11,12,13,14,15,16,24,27,28,29,35,49,70,71,72,112,113,127,129,130,131,136,151,153]
type3=[6,7,36,58]
type4=[42,54,91,100,107,109,110,114]
type5=[9,10,51,111]
type6=[8,57,81,84,125]
type7=[82,83]
type8=[46,105,115,116,117,118,128]
for i in range(157):
    for j in range(len(type1)):
        if type1[j]==i:
            type[i]=1
for i in range(157):
    for j in range(len(type2)):
        if type2[j]==i:
            type[i]=2
for i in range(157):
    for j in range(len(type3)):
        if type3[j]==i:
            type[i]=3
for i in range(157):
    for j in range(len(type4)):
        if type4[j]==i:
            type[i]=4
for i in range(157):
    for j in range(len(type5)):
        if type5[j]==i:
            type[i]=5
for i in range(157):
    for j in range(len(type6)):
        if type6[j]==i:
            type[i]=6
for i in range(157):
    for j in range(len(type7)):
        if type7[j]==i:
            type[i]=7
for i in range(157):
    for j in range(len(type8)):
        if type8[j]==i:
            type[i]=8
leixing=[None]*157
c1=0;c2=0;c3=0;c4=0;c5=0;c6=0;c7=0;c8=0
for i in range(157):
    if type[i] == 1:
        leixing[i] = c1;
        c1 += 1;
    elif type[i] == 2:
        leixing[i] = c2;
        c2 += 1;
    elif type[i] == 3:
        leixing[i] = c3;
        c3 += 1;
    elif type[i] == 4:
        leixing[i] = c4;
        c4 += 1;
    elif type[i] == 5:
        leixing[i] = c5;
        c5 += 1;
    elif type[i] == 6:
        leixing[i] = c6;
        c6 += 1;
    elif type[i] == 7:
        leixing[i] = c7;
        c6 += 1;
    elif type[i] == 8:
        leixing[i] = c8;
        c6 += 1;
a1=np.zeros((183),int);a2=np.zeros((11),int);a3=np.zeros((7),int);a4=np.zeros((4),int);a5=np.zeros((3),int);a6=np.zeros((39),int);a7=np.zeros((1),int);a8=np.zeros((1),int);a9=np.zeros((1),int);a10=np.zeros((5),int)
b1=np.zeros((183),int);b2=np.zeros((11),int);b3=np.zeros((7),int);b4=np.zeros((4),int);b5=np.zeros((3),int);b6=np.zeros((39),int);b7=np.zeros((1),int);b8=np.zeros((1),int);b9=np.zeros((1),int);b10=np.zeros((5),int)
a11=np.zeros((1),int);a12=np.zeros((9),int);a13=np.zeros((1),int);a14=np.zeros((1),int);a15=np.zeros((5),int);a16=np.zeros((1),int);a17=np.zeros((3),int);a18=np.zeros((10),int)
b11=np.zeros((1),int);b12=np.zeros((9),int);b13=np.zeros((1),int);b14=np.zeros((1),int);b15=np.zeros((5),int);b16=np.zeros((1),int);b17=np.zeros((3),int);b18=np.zeros((10),int)
a19=np.zeros((5),int);a20=np.zeros((4),int);a21=np.zeros((2),int);a22=np.zeros((4),int);a23=np.zeros((1),int);a24=np.zeros((1),int);a25=np.zeros((2),int);a27=np.zeros((5),int);a28=np.zeros((1),int);
b19=np.zeros((5),int);b20=np.zeros((4),int);b21=np.zeros((2),int);b22=np.zeros((4),int);b23=np.zeros((1),int);b24=np.zeros((1),int);b25=np.zeros((2),int);b27=np.zeros((5),int);b28=np.zeros((1),int);
a29=np.zeros((1),int);a30=np.zeros((4),int);a31=np.zeros((1),int);a32=np.zeros((100),int);
b29=np.zeros((1),int);b30=np.zeros((4),int);b31=np.zeros((1),int);b32=np.zeros((100),int);

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==1 and type[j]==1:
            a1[k]=leixing[i];b1[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==1 and type[j]==2:
            a2[k]=leixing[i];b2[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==1 and type[j]==4:
            a3[k]=leixing[i];b3[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==1 and type[j]==6:
            a4[k]=leixing[i];b4[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==1 and type[j]==8:
            a5[k]=leixing[i];b5[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==2 and type[j]==1:
            a19[k]=leixing[i];b19[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==2 and type[j]==2:
            a6[k]=leixing[i];b6[k]=leixing[j];k+=1

k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==2 and type[j]==4:
            a7[k]=leixing[i];b7[k]=leixing[j];k+=1
k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==2 and type[j]==5:
            a8[k]=leixing[i];b8[k]=leixing[j];k+=1
k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==2 and type[j]==8:
            a9[k]=leixing[i];b9[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==3 and type[j]==1:
            a20[k]=leixing[i];b20[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==3 and type[j]==2:
            a21[k]=leixing[i];b21[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==3 and type[j]==3:
            a10[k]=leixing[i];b10[k]=leixing[j];k+=1

k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==3 and type[j]==8:
            a11[k]=leixing[i];b11[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==4 and type[j]==1:
            a22[k]=leixing[i];b22[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==4 and type[j]==2:
            a23[k]=leixing[i];b23[k]=leixing[j];k+=1

k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==4 and type[j]==4:
            a12[k]=leixing[i];b12[k]=leixing[j];k+=1

k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==4 and type[j]==5:
            a13[k]=leixing[i];b13[k]=leixing[j];k+=1
k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==4 and type[j]==8:
            a14[k]=leixing[i];b14[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==5 and type[j]==1:
            a24[k]=leixing[i];b24[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==5 and type[j]==2:
            a25[k]=leixing[i];b25[k]=leixing[j];k+=1

k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==5 and type[j]==5:
            a15[k]=leixing[i];b15[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==6 and type[j]==1:
            a27[k]=leixing[i];b27[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==6 and type[j]==5:
            a28[k]=leixing[i];b28[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==6 and type[j]==7:
            a16[k]=leixing[i];b16[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==7 and type[j]==6:
            a29[k]=leixing[i];b29[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==7 and type[j]==7:
            a17[k]=leixing[i];b17[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==8 and type[j]==1:
            a30[k]=leixing[i];b30[k]=leixing[j];k+=1

k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==8 and type[j]==2:
            a31[k]=leixing[i];b31[k]=leixing[j];k+=1
# print(a30)
# print(len(b30))
k=0

for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==8 and type[j]==8:
            a18[k]=leixing[i];b18[k]=leixing[j];k+=1

def add_reverse_edges(hg, copy_ndata=True, copy_edata=True, ignore_one_type=True):#获得无向图中反过来的边

    canonical_etypes = hg.canonical_etypes
    num_nodes_dict = {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes}
    edge_dict = {}
    for etype in canonical_etypes:
        u, v = hg.edges(form='uv', order='eid', etype=etype)
        edge_dict[etype] = (u, v)
        edge_dict[(etype[2], etype[1] + '-rev', etype[0])] = (v, u)
    new_hg = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    return new_hg


g_data = {
    ('type1', 'c', 'type1'): (torch.tensor(a1, dtype=torch.int64), torch.tensor(b1, dtype=torch.int64)),
    ('type1', 'c', 'type2'): (torch.tensor(a2, dtype=torch.int64), torch.tensor(b2, dtype=torch.int64)),
    ('type1', 'c', 'type4'): (torch.tensor(a3, dtype=torch.int64), torch.tensor(b3, dtype=torch.int64)),
    ('type1', 'c', 'type6'): (torch.tensor(a4, dtype=torch.int64), torch.tensor(b4, dtype=torch.int64)),
    ('type1', 'c', 'type8'): (torch.tensor(a5, dtype=torch.int64), torch.tensor(b5, dtype=torch.int64)),
    ('type2', 'c', 'type2'): (torch.tensor(a6, dtype=torch.int64), torch.tensor(b6, dtype=torch.int64)),
    ('type2', 'c', 'type4'): (torch.tensor(a7, dtype=torch.int64), torch.tensor(b7, dtype=torch.int64)),
    ('type2', 'c', 'type5'): (torch.tensor(a8, dtype=torch.int64), torch.tensor(b8, dtype=torch.int64)),
    ('type2', 'c', 'type8'): (torch.tensor(a9, dtype=torch.int64), torch.tensor(b9, dtype=torch.int64)),
    ('type3', 'c', 'type3'): (torch.tensor(a10, dtype=torch.int64), torch.tensor(b10, dtype=torch.int64)),
    ('type3', 'c', 'type8'): (torch.tensor(a11, dtype=torch.int64), torch.tensor(b11, dtype=torch.int64)),
    ('type4', 'c', 'type4'): (torch.tensor(a12, dtype=torch.int64), torch.tensor(b12, dtype=torch.int64)),
    ('type4', 'c', 'type5'): (torch.tensor(a13, dtype=torch.int64), torch.tensor(b13, dtype=torch.int64)),
    ('type4', 'c', 'type8'): (torch.tensor(a14, dtype=torch.int64), torch.tensor(b14, dtype=torch.int64)),
    ('type5', 'c', 'type5'): (torch.tensor(a15, dtype=torch.int64), torch.tensor(b15, dtype=torch.int64)),
    ('type6', 'c', 'type7'): (torch.tensor(a16, dtype=torch.int64), torch.tensor(b16, dtype=torch.int64)),
    ('type7', 'c', 'type7'): (torch.tensor(a17, dtype=torch.int64), torch.tensor(b17, dtype=torch.int64)),
    ('type8', 'c', 'type8'): (torch.tensor(a18, dtype=torch.int64), torch.tensor(b18, dtype=torch.int64)),
    ('type2', 'c', 'type1'): (torch.tensor(a19, dtype=torch.int64), torch.tensor(b19, dtype=torch.int64)),
    ('type3', 'c', 'type1'): (torch.tensor(a20, dtype=torch.int64), torch.tensor(b20, dtype=torch.int64)),
    ('type3', 'c', 'type2'): (torch.tensor(a21, dtype=torch.int64), torch.tensor(b21, dtype=torch.int64)),
    ('type4', 'c', 'type1'): (torch.tensor(a22, dtype=torch.int64), torch.tensor(b22, dtype=torch.int64)),
    ('type4', 'c', 'type2'): (torch.tensor(a23, dtype=torch.int64), torch.tensor(b23, dtype=torch.int64)),
    ('type5', 'c', 'type1'): (torch.tensor(a24, dtype=torch.int64), torch.tensor(b24, dtype=torch.int64)),
    ('type5', 'c', 'type2'): (torch.tensor(a25, dtype=torch.int64), torch.tensor(b25, dtype=torch.int64)),
    ('type6', 'c', 'type1'): (torch.tensor(a27, dtype=torch.int64), torch.tensor(b27, dtype=torch.int64)),
    ('type6', 'c', 'type5'): (torch.tensor(a28, dtype=torch.int64), torch.tensor(b28, dtype=torch.int64)),
    ('type7', 'c', 'type6'): (torch.tensor(a29, dtype=torch.int64), torch.tensor(b29, dtype=torch.int64)),
    ('type8', 'c', 'type1'): (torch.tensor(a30, dtype=torch.int64), torch.tensor(b30, dtype=torch.int64)),
    ('type8', 'c', 'type2'): (torch.tensor(a31, dtype=torch.int64), torch.tensor(b31, dtype=torch.int64)),
}
g=dgl.heterograph(g_data)
g=add_reverse_edges(g)
print(g)
X = np.load("data/V_sub157.npy")
X1 = [];X2 = [];X3 = [];X4 = [];X5 = [];X6 = [];X7 = [];X8 = []
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
'''
def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2])]

    features = []
    for i, j in indices:
        features.append(
            X[:, :, i: i + 1].transpose(
                (0, 2, 1)))

    return torch.from_numpy(np.array(features))
'''


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

X1=generate_dataset(X1,num_timesteps_input=12,num_timesteps_output=9)
# X1 = torch.tensor(X1);X2 = torch.tensor(X2);X3 = torch.tensor(X3);X4 = torch.tensor(X4);X5 = torch.tensor(X5);X6 = torch.tensor(X6);X7 = torch.tensor(X7);X8 = torch.tensor(X8)

print(X1.shape)
XX=[]
for V in X:
    XX.append([V[0], V[1], V[2], V[3], V[4], V[5], V[17], V[18], V[19], V[20], V[21], V[22], V[23], V[25], V[26], V[30],
               V[31], V[32], V[33], V[34],
               V[37], V[38], V[39], V[40], V[41], V[43], V[44], V[45], V[47], V[48], V[50], V[52], V[53], V[55], V[56],
               V[59], V[60], V[61], V[62], V[63],
               V[64], V[65], V[66], V[67], V[68], V[69], V[73], V[74], V[75], V[76], V[77], V[78], V[79], V[80], V[85],
               V[86], V[87], V[88], V[89], V[90],
               V[92], V[93], V[94], V[95], V[96], V[97], V[98], V[99], V[101], V[102], V[103], V[104], V[106], V[108],
               V[119], V[120], V[121], V[122], V[123], V[124],
               V[126], V[132], V[133], V[134], V[135], V[137], V[138], V[139], V[140], V[141], V[142], V[143], V[144],
               V[145], V[146], V[147], V[148], V[149], V[150], V[152],
               V[154], V[155], V[156], V[11], V[12], V[13], V[14], V[15], V[16], V[24], V[27], V[28], V[29], V[35],
               V[49], V[70], V[71], V[72], V[112], V[113], V[127], V[129], V[130],
               V[131], V[136], V[151], V[153], V[6], V[7], V[36], V[25], V[42], V[54], V[91], V[100], V[107], V[109],
               V[110], V[114], V[9], V[10], V[51], V[111],
               V[8], V[57], V[81], V[84], V[125], V[82], V[83], V[46], V[105], V[115], V[116], V[117], V[118], V[128]])
XX = np.array(XX)
print(XX.shape)
split_line1 = int(XX.shape[0] * 0.6)
train_original_data = XX[ :split_line1, :]
print(train_original_data.shape)
'''
model = RGCN(in_feats=9216,hid_feats=16,out_feats=84,rel_names=g.etypes)
lfs = model(g, {'type1': X1, 'type2': X2, 'type3': X3, 'type4': X4, 'type5': X5, 'type6': X6, 'type7': X7, 'type8': X8})
te1 = lfs['type1'];te2 = lfs['type2'];te3 = lfs['type3'];te4 = lfs['type4'];te5 = lfs['type5'];te6 = lfs['type6'];te7 = lfs['type7'];te8 = lfs['type8']
te1 = te1.detach().numpy();te2 = te2.detach().numpy();te3 = te3.detach().numpy();te4 = te4.detach().numpy();te5 = te5.detach().numpy();te6 = te6.detach().numpy();te7 = te7.detach().numpy();te8 = te8.detach().numpy()
print(te1.shape)
te1 = generate_dataset(te1, num_timesteps_input=12, num_timesteps_output=9)
te2 = generate_dataset(te2, num_timesteps_input=12, num_timesteps_output=9)
te3 = generate_dataset(te3, num_timesteps_input=12, num_timesteps_output=9)
te4 = generate_dataset(te4, num_timesteps_input=12, num_timesteps_output=9)
te5 = generate_dataset(te5, num_timesteps_input=12, num_timesteps_output=9)
te6 = generate_dataset(te6, num_timesteps_input=12, num_timesteps_output=9)
te7 = generate_dataset(te7, num_timesteps_input=12, num_timesteps_output=9)
te8 = generate_dataset(te8, num_timesteps_input=12, num_timesteps_output=9)
print(te1.shape)
'''