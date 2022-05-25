import dgl
import torch as th
import networkx as nx
import numpy as np
import pandas as pd

i=0;j=0;k=0;l=0

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data
A=load_csv('w_157.csv')

for i in range(157):
    for j in range(i,157):
        if A[i][j]==1:
            A[j][i]=1
for i in range(157):
    for j in range(i):
        if A[i][j]==1:
            A[j][i]=1

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

k=0
for i in range(157):
    for j in range(i,157):
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

    edge_dict = {}
    for etype in canonical_etypes:
        u, v = hg.edges(form='uv', order='eid', etype=etype)
        edge_dict[etype] = (u, v)
        edge_dict[(etype[2], etype[1] + '-rev', etype[0])] = (v, u)
    new_hg = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict)
    return new_hg
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
