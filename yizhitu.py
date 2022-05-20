import dgl
import torch as th
import networkx as nx
import numpy as np
import pandas as pd
i=0;j=0;k=0;m=0
def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data
A=load_csv('w_157.csv')
#print(A)
for i in range(157):
    for j in range(157):
        if A[i][j]!=0:A[i][j]=1
        else :A[i][j]=0
#print(A)
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0:
            m+=1
a=[None]*m
b=[None]*m
for i in range(157):
    for j in range(i,157):
        if A[i][j]!=0:
            a[k]=i;b[k]=j;k+=1
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

#flow:车流量 speed:车速  occupancy：占用率
g_data={
     ('flow', 'distance', 'flow'): (th.tensor(a),th.tensor(b)),
     ('speed', 'distance', 'speed'): (th.tensor(a),th.tensor(b)),
     ('occupancy', 'distance', 'occupancy'): (th.tensor(a),th.tensor(b)),
    }

g=dgl.heterograph(g_data)
g1=add_reverse_edges(g)
print(g1)

