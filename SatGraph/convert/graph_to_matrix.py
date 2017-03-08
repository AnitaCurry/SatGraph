import os
import sys
import numpy as np
import scipy.sparse as sparse
import shutil
import math
import pandas as pd

def check_dest_dir(Str_DestDataPath):
    if os.path.isfile(Str_DestDataPath + '/subdata'):
        os.remove(Str_DestDataPath + '/subdata')
    if os.path.isdir(Str_DestDataPath + '/subdata'):
        shutil.rmtree(Str_DestDataPath + '/subdata')
    os.makedirs(Str_DestDataPath + '/subdata')

def csv_to_array(Str_RawData):
    p_data = pd.read_csv(Str_RawData, names=['dst', 'src'])
    t_dst_array = p_data.values[:, 0]
    t_src_array = p_data.values[:, 1]
    return t_dst_array, t_src_array

def edge_to_file(Str_DestDataPath, dst_array, src_array, par_n, shape):
    matrix_data = np.ones(len(dst_array), dtype=np.bool)
    dst_array = dst_array - par_n * shape[0]
    csr_data = (matrix_data[1:], (dst_array[1:], src_array[1:]))
    _SMat_EdgeData = sparse.csr_matrix(csr_data, shape, dtype=Dtype_All[2])

    unique1, counts1 = np.unique(src_array[1:], return_counts=True)
    vertexout = dict(zip(unique1, counts1))
    unique2, counts2 = np.unique(dst_array[1:], return_counts=True)
    vertexin = dict(zip(unique2, counts2))

    _File_PartitionData = open(
        Str_DestDataPath + '/subdata/' + str(par_n) + '.edge', 'w')
    Partition_Indices = _SMat_EdgeData.indices
    Partition_Indptr = _SMat_EdgeData.indptr
    Len_Indices = len(Partition_Indices)
    Len_Indptr = len(Partition_Indptr)
    PartitionData = np.append(len(_SMat_EdgeData.data), Len_Indices)
    PartitionData = np.append(PartitionData, Len_Indptr)
    PartitionData = np.append(PartitionData, Partition_Indices)
    PartitionData = np.append(PartitionData, Partition_Indptr)
    PartitionData = PartitionData.astype(Dtype_All[1])
    PartitionData.tofile(_File_PartitionData)
    _File_PartitionData.close()
    return vertexout, vertexin

def vertex_to_file(Str_DestDataPath, _Array_VertexOut, _Array_VertexIn):
    _File_PartitionData = open(
        Str_DestDataPath + '/subdata/' + 'vertexout', 'w')
    _Array_VertexOut.tofile(_File_PartitionData)
    _File_PartitionData.close()
    _File_PartitionData = open(
        Str_DestDataPath + '/subdata/' + 'vertexin', 'w')
    _Array_VertexIn.tofile(_File_PartitionData)
    _File_PartitionData.close()

def graph_to_matrix(Str_RawDataPath, Str_DestDataPath, Int_VertexNum, Int_PartitionNum, RawData_Num, Dtype_All):
    check_dest_dir(Str_DestDataPath)
    Int_VertexPerPartition = int(math.ceil(Int_VertexNum * 1.0 / Int_PartitionNum))
    Int_NewVertexNum = Int_PartitionNum * Int_VertexPerPartition
    _Array_VertexOut = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1])
    _Array_VertexIn = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1])

    read_id = 0
    old_data = False
    for par_n in range(Int_PartitionNum):
        dst_array = np.array([0])
        src_array = np.array([0])
        new_partition_flag = False
        while True:
            print 'read ', read_id, 'partition ', par_n
            if not old_data:
                t_dst_array, t_src_array = \
                    csv_to_array(Str_RawDataPath + '-' + str(read_id).zfill(5))
            old_data = False
            if t_dst_array[-1] < (1 + par_n) * Int_VertexPerPartition:
                split_index = np.argmax(t_dst_array >= par_n * Int_VertexPerPartition)
                dst_array = np.append(dst_array, t_dst_array[split_index:])
                src_array = np.append(src_array, t_src_array[split_index:])
                read_id = read_id + 1
            elif t_dst_array[0] >= (1 + par_n) * Int_VertexPerPartition:
                new_partition_flag = True
            else:
                new_partition_flag = True
                split_index_1 = np.argmax(t_dst_array >= (par_n) * Int_VertexPerPartition)
                split_index_2 = np.argmax(t_dst_array >= (1 + par_n) * Int_VertexPerPartition)
                dst_array = np.append(dst_array, t_dst_array[split_index_1:split_index_2])
                src_array = np.append(src_array, t_src_array[split_index_1:split_index_2])
                read_id = read_id
                old_data = True
            if new_partition_flag == True or read_id == RawData_Num:
                break
        shape=(Int_VertexPerPartition, Int_NewVertexNum)
        vertexout, vertexin = edge_to_file(Str_DestDataPath,
                                           dst_array,
                                           src_array,
                                           par_n,
                                           shape)
        for v in vertexout:
            _Array_VertexOut[v] = vertexout[v] + _Array_VertexOut[v]
        for v in vertexin:
            _Array_VertexIn[v] = vertexin[v] + _Array_VertexIn[v]

    vertex_to_file(Str_DestDataPath, _Array_VertexOut, _Array_VertexIn)
    return Str_DestDataPath + '/subdata/', Int_NewVertexNum, Int_PartitionNum, Int_VertexPerPartition

if __name__ == "__main__":
    Dtype_VertexData     = np.float32
    Dtype_VertexEdgeInfo = np.int32
    Dtype_EdgeData       = np.bool
    Dtype_All            = (Dtype_VertexData, Dtype_VertexEdgeInfo, Dtype_EdgeData)
    GraphInfo = graph_to_matrix('/Users/sunshine/Desktop/t/eu-2005-t', './', 862664, 4, 1, Dtype_All);
    print GraphInfo;
