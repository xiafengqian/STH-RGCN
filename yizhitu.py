import numpy as np
import torch
import dgl
import pandas as pd

def load_csv(path):#读csv文件，转化为数组
    data_read = pd.read_csv(path) #返回dataframe
    list = data_read.values.tolist()  #变成list
    data = np.array(list)    #数组
    #print(data.shape)
    # print(data)
    return data


i = 0;j = 0;k = 0;l = 0
A = load_csv('w_157.csv')
B = load_csv('cll.csv')

for i in range(157):
    for j in range(i, 157):
        if A[i][j] == 1:
            A[j][i] = 1
for i in range(157):
    for j in range(i):
        if A[i][j] == 1:
            A[j][i] = 1

# 将节点分为八种，分别为type1:上部密集、type2：中部密集、type3：下部密集、type4：中间低两边高、
# type5：左高、type6：右高、type7：突出一部分、type8：整体都有
type = [None] * 157
type1 = [0, 1, 2, 3, 4, 5, 17, 18, 19, 20, 21, 22, 23, 25, 26, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 43, 44, 45, 47,
         48, 50, 52, 53, 55, 56, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 73, 74, 75, 76, 77, 78, 79, 80, 85, 86, 87,
         88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 106, 108, 119, 120, 121, 122, 123, 124, 126,
         132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 152, 154, 155, 156]
type2 = [11, 12, 13, 14, 15, 16, 24, 27, 28, 29, 35, 49, 70, 71, 72, 112, 113, 127, 129, 130, 131, 136, 151, 153]
type3 = [6, 7, 36, 58]
type4 = [42, 54, 91, 100, 107, 109, 110, 114]
type5 = [9, 10, 51, 111]
type6 = [8, 57, 81, 84, 125]
type7 = [82, 83]
type8 = [46, 105, 115, 116, 117, 118, 128]
for i in range(157):
    for j in range(len(type1)):
        if type1[j] == i:
            type[i] = 1
for i in range(157):
    for j in range(len(type2)):
        if type2[j] == i:
            type[i] = 2
for i in range(157):
    for j in range(len(type3)):
        if type3[j] == i:
            type[i] = 3
for i in range(157):
    for j in range(len(type4)):
        if type4[j] == i:
            type[i] = 4
for i in range(157):
    for j in range(len(type5)):
        if type5[j] == i:
            type[i] = 5
for i in range(157):
    for j in range(len(type6)):
        if type6[j] == i:
            type[i] = 6
for i in range(157):
    for j in range(len(type7)):
        if type7[j] == i:
            type[i] = 7
for i in range(157):
    for j in range(len(type8)):
        if type8[j] == i:
            type[i] = 8
leixing = [None] * 157
c1 = 0;c2 = 0;c3 = 0;c4 = 0;c5 = 0;c6 = 0;c7 = 0;c8 = 0
for i in range(157):
    if type[i] == 1:
        leixing[i] = c1
        c1 += 1
    elif type[i] == 2:
        leixing[i] = c2
        c2 += 1
    elif type[i] == 3:
        leixing[i] = c3
        c3 += 1
    elif type[i] == 4:
        leixing[i] = c4
        c4 += 1
    elif type[i] == 5:
        leixing[i] = c5
        c5 += 1
    elif type[i] == 6:
        leixing[i] = c6
        c6 += 1
    elif type[i] == 7:
        leixing[i] = c7
        c7 += 1
    elif type[i] == 8:
        leixing[i] = c8
        c8 += 1
a1 = np.zeros((183), int);a2 = np.zeros((11), int);a3 = np.zeros((7), int);a4 = np.zeros((4), int);a5 = np.zeros((3), int)
a6 = np.zeros((39), int);a7 = np.zeros((1), int);a8 = np.zeros((1), int);a9 = np.zeros((1), int);a10 = np.zeros((5), int)
b1 = np.zeros((183), int);b2 = np.zeros((11), int);b3 = np.zeros((7), int);b4 = np.zeros((4), int);b5 = np.zeros((3), int)
b6 = np.zeros((39), int);b7 = np.zeros((1), int);b8 = np.zeros((1), int);b9 = np.zeros((1), int);b10 = np.zeros((5), int)
a11 = np.zeros((1), int);a12 = np.zeros((9), int);a13 = np.zeros((1), int);a14 = np.zeros((1), int);a15 = np.zeros((5), int)
a16 = np.zeros((1), int);a17 = np.zeros((3), int);a18 = np.zeros((10), int)
b11 = np.zeros((1), int);b12 = np.zeros((9), int);b13 = np.zeros((1), int);b14 = np.zeros((1), int);b15 = np.zeros((5), int)
b16 = np.zeros((1), int);
b17 = np.zeros((3), int);b18 = np.zeros((10), int)
a19 = np.zeros((5), int);a20 = np.zeros((4), int);a21 = np.zeros((2), int);a22 = np.zeros((4), int);a23 = np.zeros((1), int);a24 = np.zeros((1), int);
a25 = np.zeros((2), int);a27 = np.zeros((5), int);a28 = np.zeros((1), int);
b19 = np.zeros((5), int);b20 = np.zeros((4), int);b21 = np.zeros((2), int);b22 = np.zeros((4), int);b23 = np.zeros((1), int);b24 = np.zeros((1), int);
b25 = np.zeros((2), int);b27 = np.zeros((5), int);b28 = np.zeros((1), int);a29 = np.zeros((1), int);a30 = np.zeros((4), int);
a31 = np.zeros((1), int);a32 = np.zeros((100), int);
b29 = np.zeros((1), int);b30 = np.zeros((4), int);b31 = np.zeros((1), int);b32 = np.zeros((100), int);
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 1 and type[j] == 1:
            a1[k] = leixing[i];
            b1[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 1 and type[j] == 2:
            a2[k] = leixing[i];
            b2[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 1 and type[j] == 4:
            a3[k] = leixing[i];
            b3[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 1 and type[j] == 6:
            a4[k] = leixing[i];
            b4[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 1 and type[j] == 8:
            a5[k] = leixing[i];
            b5[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 2 and type[j] == 1:
            a19[k] = leixing[i];
            b19[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 2 and type[j] == 2:
            a6[k] = leixing[i];
            b6[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 2 and type[j] == 4:
            a7[k] = leixing[i];
            b7[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 2 and type[j] == 5:
            a8[k] = leixing[i];
            b8[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 2 and type[j] == 8:
            a9[k] = leixing[i];
            b9[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 3 and type[j] == 1:
            a20[k] = leixing[i];
            b20[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 3 and type[j] == 2:
            a21[k] = leixing[i];
            b21[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 3 and type[j] == 3:
            a10[k] = leixing[i];
            b10[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 3 and type[j] == 8:
            a11[k] = leixing[i];
            b11[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 4 and type[j] == 1:
            a22[k] = leixing[i];
            b22[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 4 and type[j] == 2:
            a23[k] = leixing[i];
            b23[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 4 and type[j] == 4:
            a12[k] = leixing[i];
            b12[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 4 and type[j] == 5:
            a13[k] = leixing[i];
            b13[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 4 and type[j] == 8:
            a14[k] = leixing[i];
            b14[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 5 and type[j] == 1:
            a24[k] = leixing[i];
            b24[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 5 and type[j] == 2:
            a25[k] = leixing[i];
            b25[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 5 and type[j] == 5:
            a15[k] = leixing[i];
            b15[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 6 and type[j] == 1:
            a27[k] = leixing[i];
            b27[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 6 and type[j] == 5:
            a28[k] = leixing[i];
            b28[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 6 and type[j] == 7:
            a16[k] = leixing[i];
            b16[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 7 and type[j] == 6:
            a29[k] = leixing[i];
            b29[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 7 and type[j] == 7:
            a17[k] = leixing[i];
            b17[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 8 and type[j] == 1:
            a30[k] = leixing[i];
            b30[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 8 and type[j] == 2:
            a31[k] = leixing[i];
            b31[k] = leixing[j];
            k += 1
k = 0
for i in range(157):
    for j in range(i, 157):
        if A[i][j] != 0 and type[i] == 8 and type[j] == 8:
            a18[k] = leixing[i];
            b18[k] = leixing[j];
            k += 1


def add_reverse_edges(hg, copy_ndata=True, copy_edata=True, ignore_one_type=True):
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
g = dgl.heterograph(g_data)
g = add_reverse_edges(g)
print(g)
dgl.save_graphs("demo_graph.bin", g)