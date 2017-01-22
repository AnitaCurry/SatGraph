import numpy as np

DATAPATH = '/home/mapred/GraphData/twitter/subdata/'
OUTPATH  = '/home/mapred/GraphData/twitter/edge/'
P_NUM    = 50
EDGE_NUM = 30000000

def load_edgedata(PartitionID):
    edge_path = DATAPATH + str(PartitionID) + '.edge'
    _file = open(edge_path, 'r')
    temp = np.fromfile(_file, np.int32)
    edge_num = data = temp[0]
    indices = temp[3:3 + int(temp[1])]
    indptr = temp[3 + int(temp[1]):3 + int(temp[1]) + int(temp[2])]
    _file.close()
    return edge_num, indices, indptr

mat_1 = ();
mat_2 = ();
mat_1 = load_edgedata(0)
p_id = 0

for i in range (1:P_NUM):
    if mat_1[0] < EDGE_NUM:
        mat_2 = load_edgedata(i)
        mat_1[0] += mat_2[0]
        mat_1[1]  = np.append(mat_1[1], mat_2[1])
        mat_2[2] += mat_1[2][-1]
        mat_1[2]  = np.append(mat_1[2], mat_2[2][1:])
