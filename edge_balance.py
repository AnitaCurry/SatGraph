import numpy as np
import scipy.sparse as sparse

DATAPATH = '/home/mapred/GraphData/twitter/subdata/'
OUTPATH  = '/home/mapred/GraphData/twitter/edge/'
P_NUM    = 50
EDGE_NUM = 30000000


def load_edgedata(PartitionID):
    edge_path = DATAPATH + str(PartitionID) + '.edge'
    _file = open(edge_path, 'r')
    temp = np.fromfile(_file, np.int32)
    edge_num = temp[0]
    indices = temp[3:3 + int(temp[1])]
    indptr = temp[3 + int(temp[1]):3 + int(temp[1]) + int(temp[2])]
    _file.close()
    return edge_num, indices, indptr

mat_1 = ();
mat_2 = ();
mat_1 = list(load_edgedata(0))
p_id = 0
v_id = 0
i = 1

while 1:
    if mat_1[0] < EDGE_NUM:
        if i == P_NUM:
            break
        mat_2 = list(load_edgedata(i))
        mat_1[0] += mat_2[0]
        mat_1[1]  = np.append(mat_1[1], mat_2[1])
        mat_2[2] += mat_1[2][-1]
        mat_1[2]  = np.append(mat_1[2], mat_2[2][1:])
        i = i + 1
    else:
        start_id = v_id
        s_row = np.argmax(mat_1[2]>=EDGE_NUM)
        end_id = start_id + s_row
        mat = []
        mat.append(mat_1[2][s_row])
        t = mat_1[2][s_row]
        mat.append(mat_1[1][:t])
        mat.append(mat_1[2][:s_row+1])
        print "Partition:", p_id, '\t From:', start_id, '\t to:', end_id
        _File_PartitionData = open(
            OUTPATH + str(p_id) + '.edge', 'w')
        Partition_Indices = mat[1]
        Partition_Indptr = mat[2]
        Len_Indices = len(Partition_Indices)
        Len_Indptr = len(Partition_Indptr)
        PartitionData = np.append(mat[0], Len_Indices)
        PartitionData = np.append(PartitionData, Len_Indptr)
        PartitionData = np.append(PartitionData, start_id)
        PartitionData = np.append(PartitionData, end_id)
        PartitionData = np.append(PartitionData, Partition_Indices)
        PartitionData = np.append(PartitionData, Partition_Indptr)
        PartitionData = PartitionData.astype(np.int32)
        PartitionData.tofile(_File_PartitionData)
        _File_PartitionData.close()



        mat_1[0] -= mat[0]
        mat_1[1]  = mat_1[1][t:]
        mat_1[2]  = mat_1[2][s_row:]
        mat_1[2]  = mat_1[2] - mat_1[2][0]
        # print mat_1[0], len(mat_1[1]), mat_1[2][-1]
        p_id += 1
        v_id  = end_id

start_id = v_id
end_id   = start_id + len(mat_1[2]) - 1
print "Partition:", p_id, '\t From:', start_id, '\t to:', end_id
