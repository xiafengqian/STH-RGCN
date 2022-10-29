import dgl
import torch as th
import networkx as nx
import numpy as np
import pandas as pd
<<<<<<< HEAD

=======
from openhgnn import Experiment
from openhgnn.dataset import AsLinkPredictionDataset, generate_random_hg, AsTimePredictionDataset
from dgl import transforms as T
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset
from dgl.dataloading.negative_sampler import GlobalUniform
>>>>>>> 0385d17 (STGCNRGCN)
i=0;j=0;k=0;l=0

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data
A=load_csv('w_157.csv')
<<<<<<< HEAD
=======
B=load_csv('cll.csv')
>>>>>>> 0385d17 (STGCNRGCN)

for i in range(157):
    for j in range(i,157):
        if A[i][j]==1:
            A[j][i]=1
for i in range(157):
    for j in range(i):
        if A[i][j]==1:
            A[j][i]=1

<<<<<<< HEAD
#print(A)
df = pd.read_csv("information.csv")
#print(df)
lanes=df['Lanes']
lanes=np.array(lanes)
#print(lanes)

type=[None]*157
for i in range(157):
    if df.iloc[i]['Type']=='ML':
        type[i]=0
    elif df.iloc[i]['Type']=='HV':
        type[i]=1
#print(type)
chedao=[None]*157
leixing=[None]*157
c1=0;c2=0;c3=0;c4=0;c5=0;c6=0
for i in range(157):
    if lanes[i] == 1:
        chedao[i] = c1;
        c1 += 1;
    elif lanes[i] == 2:
        chedao[i] = c2;
        c2 += 1;
    elif lanes[i] == 3:
        chedao[i] = c3;
        c3 += 1;
    elif lanes[i] == 4:
        chedao[i] = c4;
        c4 += 1;
    elif lanes[i] == 5:
        chedao[i] = c5;
        c5 += 1;
    elif lanes[i] == 6:
        chedao[i] = c6;
        c6 += 1;
t0=0;t1=0
for i in range(157):
    if type[i] == 0:
        leixing[i] = t0;
        t0 += 1;
    elif type[i] == 1:
        leixing[i] = t1;
        t1 += 1;
#print(chedao)
#print(leixing)
#chedao=[0,1,2,3,4,0,5,0,1,2,6,0,1,2,3,7,8,9,0,10,11,12,13,14,15,16,17,3,18,19,4,1,20,21,22,23,4,2,5,5,21,25,26,27,3,4,6,28,29,30,31,5,32,33,7,34,35,8,1,36,6,7,8,37,38,39,9,10,11,6,12,7,40,13,14,41,42,43,44,45,46,9,10,11,12,47,48,49,50,51,52,53,15,16,17,18,54,19,20,21,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,8,9,22,72,73,23,24,74,13,75,76,0,10,77,14,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102]
#leixing=[0,1,2,3,4,5,6,0,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,1,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,2,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153]
a1=np.zeros((4),int);a2=np.zeros((2),int);a3=np.zeros((6),int);a4=np.zeros((17),int);a5=np.zeros((9),int);a6=np.zeros((1),int);a10=np.zeros((122),int);a11=np.zeros((12),int);a12=np.zeros((2),int);a15=np.zeros((12),int);a16=np.zeros((6),int);a21=np.zeros((8),int);a8=np.zeros((3),int);a7=np.zeros((1),int);a9=np.zeros((11),int);a13=np.zeros((4),int);a14=np.zeros((13),int);a17=np.zeros((1),int);a18=np.zeros((1),int);a19=np.zeros((5),int);a20=np.zeros((2),int);a22=np.zeros((231),int);a23=np.zeros((5),int);a24=np.zeros((6),int);
b1=np.zeros((4),int);b2=np.zeros((2),int);b3=np.zeros((6),int);b4=np.zeros((17),int);b5=np.zeros((9),int);b6=np.zeros((1),int);b10=np.zeros((122),int);b11=np.zeros((12),int);b12=np.zeros((2),int);b15=np.zeros((12),int);b16=np.zeros((6),int);b21=np.zeros((8),int);b8=np.zeros((3),int);b7=np.zeros((1),int);b9=np.zeros((11),int);b13=np.zeros((4),int);b14=np.zeros((13),int);b17=np.zeros((1),int);b18=np.zeros((1),int);b19=np.zeros((5),int);b20=np.zeros((2),int);b22=np.zeros((231),int);b23=np.zeros((5),int);b24=np.zeros((6),int);
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==1 and lanes[j]==4:
            a1[k]=chedao[i];b1[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==1 and lanes[j]==6:
            a2[k]=chedao[i];b2[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==3 and lanes[j]==3:
            a3[k]=chedao[i];b3[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==3 and lanes[j]==4:
            a4[k]=chedao[i];b4[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==3 and lanes[j]==5:
            a5[k]=chedao[i];b5[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==3 and lanes[j]==6:
            a6[k]=chedao[i];b6[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==4 and lanes[j]==1:
            a7[k]=chedao[i];b7[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==4 and lanes[j]==2:
            a8[k]=chedao[i];b8[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==4 and lanes[j]==3:
            a9[k]=chedao[i];b9[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==4 and lanes[j]==4:
            a10[k]=chedao[i];b10[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==4 and lanes[j]==5:
            a11[k]=chedao[i];b11[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==4 and lanes[j]==6:
            a12[k]=chedao[i];b12[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==5 and lanes[j]==3:
            a13[k]=chedao[i];b13[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==5 and lanes[j]==4:
            a14[k]=chedao[i];b14[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==5 and lanes[j]==5:
            a15[k]=chedao[i];b15[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==5 and lanes[j]==6:
            a16[k]=chedao[i];b16[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==6 and lanes[j]==2:
            a17[k]=chedao[i];b17[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==6 and lanes[j]==3:
            a18[k]=chedao[i];b18[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==6 and lanes[j]==4:
            a19[k]=chedao[i];b19[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==6 and lanes[j]==5:
            a20[k]=chedao[i];b20[k]=chedao[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and lanes[i]==6 and lanes[j]==6:
            a21[k]=chedao[i];b21[k]=chedao[j];k+=1
=======
def matrix_multiplication(mat1, mat2):  # 二维矩阵乘法
    s1, s2 = np.array(mat1).shape, np.array(mat2).shape
    if len(s1) != 2 and len(s2) != 2:
        print("矩阵维度错误，请重新输入")
        return -1
    mat1_row, mat1_col, mat2_row, mat2_col = len(mat1), len(mat1[0]), len(mat2), len(mat2[0])
    if mat1_col != mat2_row:
        print("矩阵1的列不等于矩阵2的行，无法进行乘法！")
        return -1
    new_mat_row, new_mat_col = mat1_row, mat2_col
    new_mat = [[0 for i in range(new_mat_col)] for i in range(new_mat_row)]
    for i in range(new_mat_row):
        for j in range(new_mat_col):
            add = 0
            for k in range(mat1_col):
                add += mat1[i][k] * mat2[k][j]
            new_mat[i][j] = add
    return new_mat

# C=matrix_multiplication(B,A)
# C:t*1
# C=np.array(C)

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
>>>>>>> 0385d17 (STGCNRGCN)

k=0
for i in range(157):
    for j in range(i,157):
<<<<<<< HEAD
        if A[i][j]!=0 and type[i]==0 and type[j]==0:
            a22[k]=leixing[i];b22[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==0 and type[j]==1:
            a23[k]=leixing[i];b23[k]=leixing[j];k+=1
k=0
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0 and type[i]==1 and type[j]==0:
            a24[k]=leixing[i];b24[k]=leixing[j];k+=1

def add_reverse_edges(hg, copy_ndata=True, copy_edata=True, ignore_one_type=True):#获得无向图中反过来的边
    # get node cnt for each ntype

    canonical_etypes = hg.canonical_etypes
    num_nodes_dict = {ntype: hg.number_of_nodes(ntype) for ntype in hg.ntypes}

=======
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
>>>>>>> 0385d17 (STGCNRGCN)
    edge_dict = {}
    for etype in canonical_etypes:
        u, v = hg.edges(form='uv', order='eid', etype=etype)
        edge_dict[etype] = (u, v)
        edge_dict[(etype[2], etype[1] + '-rev', etype[0])] = (v, u)
    new_hg = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    return new_hg
<<<<<<< HEAD
#flow:车流量 speed:车速  occupancy：占用率 lanes:车道 type:道路类型
g_data={
     ('lanes1', 'd', 'lanes4'):(th.tensor(a1),th.tensor(b1)),
     ('lanes1', 'd', 'lanes6'):(th.tensor(a2),th.tensor(b2)),
     ('lanes3', 'd', 'lanes3'):(th.tensor(a3),th.tensor(b3)),
     ('lanes3', 'd', 'lanes4'):(th.tensor(a4),th.tensor(b4)),
     ('lanes3', 'd', 'lanes5'):(th.tensor(a5),th.tensor(b5)),
     ('lanes3', 'd', 'lanes6'):(th.tensor(a6),th.tensor(b6)),
     ('lanes4', 'd', 'lanes1'):(th.tensor(a7),th.tensor(b7)),
     ('lanes4', 'd', 'lanes2'):(th.tensor(a8),th.tensor(b8)),
     ('lanes4', 'd', 'lanes3'):(th.tensor(a9),th.tensor(b9)),
     ('lanes4', 'd', 'lanes4'):(th.tensor(a10),th.tensor(b10)),
     ('lanes4', 'd', 'lanes5'):(th.tensor(a11),th.tensor(b11)),
     ('lanes4', 'd', 'lanes6'):(th.tensor(a12),th.tensor(b12)),
     ('lanes5', 'd', 'lanes3'):(th.tensor(a13),th.tensor(b13)),
     ('lanes5', 'd', 'lanes4'):(th.tensor(a14),th.tensor(b14)),
     ('lanes5', 'd', 'lanes5'):(th.tensor(a15),th.tensor(b15)),
     ('lanes5', 'd', 'lanes6'):(th.tensor(a16),th.tensor(b16)),
     ('lanes6', 'd', 'lanes2'):(th.tensor(a17),th.tensor(b17)),
     ('lanes6', 'd', 'lanes3'):(th.tensor(a18),th.tensor(b18)),
     ('lanes6', 'd', 'lanes4'):(th.tensor(a19),th.tensor(b19)),
     ('lanes6', 'd', 'lanes5'):(th.tensor(a20),th.tensor(b20)),
     ('lanes6', 'd', 'lanes6'):(th.tensor(a21),th.tensor(b21)),
     ('ML', 'd', 'ML'): (th.tensor(a22),th.tensor(b22)),
     ('ML', 'd', 'HV'): (th.tensor(a23),th.tensor(b23)),
     ('HV', 'd', 'ML'): (th.tensor(a24),th.tensor(b24)),
    }
g=dgl.heterograph(g_data)
g1=add_reverse_edges(g)
print(g1)
=======

g_data={
    ('type1', 'c', 'type1'):(th.tensor(a1,dtype=th.int64),th.tensor(b1,dtype=th.int64)),
    ('type1', 'c', 'type2'):(th.tensor(a2,dtype=th.int64),th.tensor(b2,dtype=th.int64)),
    ('type1', 'c', 'type4'):(th.tensor(a3,dtype=th.int64),th.tensor(b3,dtype=th.int64)),
    ('type1', 'c', 'type6'):(th.tensor(a4,dtype=th.int64),th.tensor(b4,dtype=th.int64)),
    ('type1', 'c', 'type8'):(th.tensor(a5,dtype=th.int64),th.tensor(b5,dtype=th.int64)),
    ('type2', 'c', 'type2'):(th.tensor(a6,dtype=th.int64),th.tensor(b6,dtype=th.int64)),
    ('type2', 'c', 'type4'):(th.tensor(a7,dtype=th.int64),th.tensor(b7,dtype=th.int64)),
    ('type2', 'c', 'type5'):(th.tensor(a8,dtype=th.int64),th.tensor(b8,dtype=th.int64)),
    ('type2', 'c', 'type8'):(th.tensor(a9,dtype=th.int64),th.tensor(b9,dtype=th.int64)),
    ('type3', 'c', 'type3'):(th.tensor(a10,dtype=th.int64),th.tensor(b10,dtype=th.int64)),
    ('type3', 'c', 'type8'):(th.tensor(a11,dtype=th.int64),th.tensor(b11,dtype=th.int64)),
    ('type4', 'c', 'type4'):(th.tensor(a12,dtype=th.int64),th.tensor(b12,dtype=th.int64)),
    ('type4', 'c', 'type5'):(th.tensor(a13,dtype=th.int64),th.tensor(b13,dtype=th.int64)),
    ('type4', 'c', 'type8'):(th.tensor(a14,dtype=th.int64),th.tensor(b14,dtype=th.int64)),
    ('type5', 'c', 'type5'):(th.tensor(a15,dtype=th.int64),th.tensor(b15,dtype=th.int64)),
    ('type6', 'c', 'type7'):(th.tensor(a16,dtype=th.int64),th.tensor(b16,dtype=th.int64)),
    ('type7', 'c', 'type7'):(th.tensor(a17,dtype=th.int64),th.tensor(b17,dtype=th.int64)),
    ('type8', 'c', 'type8'):(th.tensor(a18,dtype=th.int64),th.tensor(b18,dtype=th.int64)),
    ('type2', 'c', 'type1'):(th.tensor(a19,dtype=th.int64),th.tensor(b19,dtype=th.int64)),
    ('type3', 'c', 'type1'):(th.tensor(a20,dtype=th.int64),th.tensor(b20,dtype=th.int64)),
    ('type3', 'c', 'type2'):(th.tensor(a21,dtype=th.int64),th.tensor(b21,dtype=th.int64)),
    ('type4', 'c', 'type1'):(th.tensor(a22,dtype=th.int64),th.tensor(b22,dtype=th.int64)),
    ('type4', 'c', 'type2'):(th.tensor(a23,dtype=th.int64),th.tensor(b23,dtype=th.int64)),
    ('type5', 'c', 'type1'):(th.tensor(a24,dtype=th.int64),th.tensor(b24,dtype=th.int64)),
    ('type5', 'c', 'type2'):(th.tensor(a25,dtype=th.int64),th.tensor(b25,dtype=th.int64)),
    ('type6', 'c', 'type1'):(th.tensor(a27,dtype=th.int64),th.tensor(b27,dtype=th.int64)),
    ('type6', 'c', 'type5'):(th.tensor(a28,dtype=th.int64),th.tensor(b28,dtype=th.int64)),
    ('type7', 'c', 'type6'):(th.tensor(a29,dtype=th.int64),th.tensor(b29,dtype=th.int64)),
    ('type8', 'c', 'type1'):(th.tensor(a30,dtype=th.int64),th.tensor(b30,dtype=th.int64)),
    ('type8', 'c', 'type2'):(th.tensor(a31,dtype=th.int64),th.tensor(b31,dtype=th.int64)),
}
g=dgl.heterograph(g_data)
g=add_reverse_edges(g)

n_type1 = 103
time_step = 9215
num_for_predict = 12
batch_size=64
device=th.device('cpu')
category='type1'

meta_paths_dict = {'APA': [('author', 'author-paper', 'paper'), ('paper', 'rev_author-paper', 'author')]}
target_link = [('author', 'author-paper', 'paper')]
target_link_r = [('paper', 'rev_author-paper', 'author')]


class MyTPDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='my-tp-dataset', force_reload=True)

    def process(self):
        # Generate a random heterogeneous graph with labels on target node type.
        hg = g
        self._g = hg
        print(hg.idtype)
        print(hg)

    # Some models require meta paths, you can set meta path dict for this dataset.
    @property
    def meta_paths_dict(self):
        return meta_paths_dict

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1


class MySplitTPDatasetWithNegEdges(DGLDataset):
    def __init__(self):
        super().__init__(name='my-split-tp-dataset-with-neg-edges',
                         force_reload=True)

    def process(self):
        hg = g

        # Optionally you can specify the split masks on your own, otherwise we will
        # automatically split them randomly by the input split ratio to AsLinkPredictionDataset.
        split_ratio = [0.8, 0.1, 0.1]
        for etype in target_link:
            num = hg.num_edges(etype)
            train_mask = th.zeros(num).bool()
            train_mask[0: int(split_ratio[0] * num)] = True
            val_mask = th.zeros(num).bool()
            val_mask[int(split_ratio[0] * num): int((split_ratio[0] + split_ratio[1]) * num)] = True
            test_mask = th.zeros(num).bool()
            test_mask[int((split_ratio[0] + split_ratio[1]) * num):] = True
            hg.edges[etype].data['train_mask'] = train_mask
            hg.edges[etype].data['val_mask'] = val_mask
            hg.edges[etype].data['test_mask'] = test_mask
        # Furthermore, you can also optionally sample the negative edges and process them as
        # properties neg_val_edges and neg_test_edges. We will first check whether the dataset
        # has properties named neg_val_edges and neg_test_edges. If no, we will sample negative
        # val/test edges according to neg_ratio and neg_sampler.
        self._neg_val_edges, self._neg_test_edges = self._sample_negative_edges(hg)
        self._g = hg

    def __getitem__(self, idx):
        return self._g

    def __len__(self):
        return 1



def train_with_custom_lp_dataset(dataset):
    experiment = Experiment(model='RGCN', dataset=dataset, task='time_prediction', gpu=-1)
    experiment.run()


if __name__ == '__main__':
    myTPDataset = AsTimePredictionDataset(MyTPDataset(), target_link=target_link, target_ntype=category,
                                          split_ratio=[0.8, 0.1, 0.1], force_reload=True)
    train_with_custom_lp_dataset(myTPDataset)

    mySplitTPDatasetWithNegEdges = AsTimePredictionDataset(MySplitTPDatasetWithNegEdges(), target_ntype=category,
                                                           split_ratio=[0.8, 0.1, 0.1],force_reload=True)
    train_with_custom_lp_dataset(mySplitTPDatasetWithNegEdges)
>>>>>>> 0385d17 (STGCNRGCN)
