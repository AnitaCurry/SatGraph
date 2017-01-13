'''
Created on 14 Apr 2016

@author: sunshine
'''
import os
import sys
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI
import shutil
import math
import threading
import Queue
import zmq
from time import sleep
import snappy
from Cython.Plex.Regexps import Empty
from numpy import linalg as LA
# from scipy.sparse import coo_matrix
import pandas as pd

QueueUpdatedVertex = Queue.Queue()


def preprocess_graph(Str_RawDataPath, Str_DestDataPath, Str_Seq='\t'):
  if not os.path.isfile(Str_RawDataPath):
    return -1
  _File_RawData   = open(Str_RawDataPath, 'r')
  _Dict_Map       = {} #can also be levelDB
  _Processed_Data = []
  _Int_InitialNum = -1
  while True:
    _Str_Line = _File_RawData.readline()
    if len(_Str_Line) == 0:
      break
    if _Str_Line[0] == '#':
      continue
    _Str_Temp = _Str_Line.split(Str_Seq)
    if len(_Str_Temp) != 2:
      continue
    try:
      _Int_i        = int(_Str_Temp[0])
      _Int_j        = int(_Str_Temp[1])
      _Int_Mapped_i = 0
      _Int_Mapped_j = 0
      if _Int_i in _Dict_Map:
        _Int_Mapped_i = _Dict_Map[_Int_i]
      else:
        _Int_InitialNum   = _Int_InitialNum + 1
        _Dict_Map[_Int_i] = _Int_InitialNum
        _Int_Mapped_i     = _Int_InitialNum

      if _Int_j in _Dict_Map:
        _Int_Mapped_j = _Dict_Map[_Int_j]
      else:
        _Int_InitialNum   = _Int_InitialNum + 1
        _Dict_Map[_Int_j] = _Int_InitialNum
        _Int_Mapped_j     = _Int_InitialNum
      _Processed_Data.append((_Int_Mapped_i, _Int_Mapped_j))
    except:
      print 'Cannot format Data ', _Str_Line
  _File_RawData.close()
  _File_DestData = open(Str_DestDataPath, 'w')
  for i in _Processed_Data:
    _File_DestData.write(str(i[0]) + '\t' + str(i[1]) + '\n')
  _File_DestData.close()
  return len(_Dict_Map), len(_Processed_Data)

# def graph_to_matrix(Str_RawDataPath, Str_DestDataPath, Int_VertexNum, Int_PartitionNum, Dtype_All, Str_Seq=','):
#   if not os.path.isfile(Str_RawDataPath):
#     return -1
#
#   Int_VertexPerPartition = int(math.ceil(Int_VertexNum * 1.0 / Int_PartitionNum))
#   Int_NewVertexNum       = Int_PartitionNum * Int_VertexPerPartition
#   _SMat_EdgeData         = sparse.lil_matrix((Int_NewVertexNum, Int_NewVertexNum), dtype=Dtype_All[2])
#   print 'initial edge matrix';
#   # _Array_VertexIn        = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1])
#   _Array_VertexOut       = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1])
#   print 'initial vertex matrix';
#   _File_RawData          = open(Str_RawDataPath, 'r')
#   _Str_Line              = ''
#   print Int_NewVertexNum, Int_VertexPerPartition;
#
#   read_edge = 0;
#   #Read data from the raw data file
#   while True:
#     if read_edge % 100000 == 0:
#       #print read_edge*1.0/101355853, '#', read_edge;
#       print read_edge*1.0/91792261600, '#', read_edge;
#     read_edge = read_edge + 1
#
#     _Str_Line = _File_RawData.readline()
#     if len(_Str_Line) == 0:
#       break
#     #if _Str_Line[0] == '#':
#     #  continue
#     _Str_Temp = _Str_Line.split(Str_Seq)
#     #if len(_Str_Temp) != 2:
#     #  continue
#     #try:
#     _Int_i = int(_Str_Temp[0])
#     _Int_j = int(_Str_Temp[1])
#     #except:
#     #  print 'Cannot format Data ', _Str_Line
#     #j is dst, i is src
#     _SMat_EdgeData[_Int_j, _Int_i] = 1
#     # _Array_VertexIn[_Int_j]        = _Array_VertexIn[_Int_j] + 1
#     _Array_VertexOut[_Int_i]       = _Array_VertexOut[_Int_i] + 1
#   _File_RawData.close()
#
#   _Array_VertexOut[np.where(_Array_VertexOut == 0)] = 1
#   # _Array_VertexIn[np.where(_Array_VertexIn == 0)] = 1
#   _SMat_EdgeData = _SMat_EdgeData.tocsr()
#   if os.path.isfile(Str_DestDataPath + '/subdata'):
#     os.remove(Str_DestDataPath + '/subdata')
#   if os.path.isdir(Str_DestDataPath + '/subdata'):
#     shutil.rmtree(Str_DestDataPath + '/subdata')
#   os.makedirs(Str_DestDataPath + '/subdata')
#
#   for i in range(Int_PartitionNum):
#     _File_PartitionData     = open(Str_DestDataPath + '/subdata/' + str(i) + '.edge', 'w')
#     Partition_SMat_EdgeData = _SMat_EdgeData[i * Int_VertexPerPartition:(i + 1) * Int_VertexPerPartition]
#     Partition_Indices       = Partition_SMat_EdgeData.indices
#     Partition_Indptr        = Partition_SMat_EdgeData.indptr
#     Len_Indices             = len(Partition_Indices)
#     Len_Indptr              = len(Partition_Indptr)
#     PartitionData           = np.append(len(Partition_SMat_EdgeData.data), Len_Indices)
#     PartitionData           = np.append(PartitionData, Len_Indptr)
#     PartitionData           = np.append(PartitionData, Partition_Indices)
#     PartitionData           = np.append(PartitionData, Partition_Indptr)
#     PartitionData           = PartitionData.astype(Dtype_All[1])
#     PartitionData.tofile(_File_PartitionData)
#     _File_PartitionData.close()
#
#   _File_PartitionData = open(Str_DestDataPath + '/subdata/' + 'vertexout', 'w')
#   _Array_VertexOut.tofile(_File_PartitionData)
#   _File_PartitionData.close()
#   # _File_PartitionData = open(Str_DestDataPath + '/subdata/' + 'vertexin', 'w')
#   # _Array_VertexIn.tofile(_File_PartitionData)
#   # _File_PartitionData.close()
#   return Str_DestDataPath + '/subdata/', Int_NewVertexNum, Int_PartitionNum, Int_VertexPerPartition


def graph_to_matrix(Str_RawDataPath, Str_DestDataPath, Int_VertexNum, Int_PartitionNum, Sub_Num, Dtype_All, Str_Seq=','):
  # if not os.path.isfile(Str_RawDataPath):
  #   return -1
  if os.path.isfile(Str_DestDataPath + '/subdata'):
    os.remove(Str_DestDataPath + '/subdata')
  if os.path.isdir(Str_DestDataPath + '/subdata'):
    shutil.rmtree(Str_DestDataPath + '/subdata')
  os.makedirs(Str_DestDataPath + '/subdata')

  Int_VertexPerPartition = int(math.ceil(Int_VertexNum * 1.0 / Int_PartitionNum))
  Int_NewVertexNum       = Int_PartitionNum * Int_VertexPerPartition
  _Array_VertexOut       = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1])
  _Array_VertexIn        = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1])

  read_id = 0;
  # read_rows = 0;

  for par_n in range(Int_PartitionNum):
    dst_array = np.array([0])
    src_array = np.array([0])
    new_partition_flag = False;
    while True:
      print 'read ', read_id, 'partition ', par_n;
      p_data      = pd.read_csv(Str_RawDataPath+'-'+str(read_id).zfill(5), names=['dst', 'src'])
      t_dst_array = p_data.values[:,0]
      t_src_array = p_data.values[:,1]
      if t_dst_array[-1] < (1+par_n)*Int_VertexPerPartition:
        split_index = np.argmax(t_dst_array >= (par_n)*Int_VertexPerPartition)
        dst_array = np.append(dst_array, t_dst_array[split_index:])
        src_array = np.append(src_array, t_src_array[split_index:])
        read_id = read_id + 1
      elif t_dst_array[0] >= (1+par_n)*Int_VertexPerPartition:
        new_partition_flag = True
      else:
        new_partition_flag = True
        split_index_1  = np.argmax(t_dst_array >= (par_n)*Int_VertexPerPartition)
        split_index_2 = np.argmax(t_dst_array >= (1+par_n)*Int_VertexPerPartition)
        dst_array = np.append(dst_array, t_dst_array[split_index_1:split_index_2])
        src_array = np.append(src_array, t_src_array[split_index_1:split_index_2])
        read_id = read_id
      if new_partition_flag == True or read_id == Sub_Num:
        break
    data = np.ones(len(dst_array), dtype=np.bool)
    dst_array = dst_array - par_n * Int_VertexPerPartition
    _SMat_EdgeData = sparse.csr_matrix((data[1:], (dst_array[1:], src_array[1:])), shape=(Int_VertexPerPartition, Int_NewVertexNum), dtype=Dtype_All[2])

    unique, counts = np.unique(src_array[1:], return_counts=True)
    vertexout = dict(zip(unique, counts))
    for v in vertexout:
      _Array_VertexOut[v] = vertexout[v] + _Array_VertexOut[v];

    unique, counts = np.unique(dst_array[1:], return_counts=True)
    vertexin = dict(zip(unique, counts))
    for v in vertexin:
      _Array_VertexIn[v] = vertexin[v] + _Array_VertexIn[v];

    _File_PartitionData     = open(Str_DestDataPath + '/subdata/' + str(par_n) + '.edge', 'w')
    # Partition_SMat_EdgeData = _SMat_EdgeData[par_n * Int_VertexPerPartition : (par_n + 1) * Int_VertexPerPartition]
    Partition_Indices       = _SMat_EdgeData.indices
    Partition_Indptr        = _SMat_EdgeData.indptr
    Len_Indices             = len(Partition_Indices)
    Len_Indptr              = len(Partition_Indptr)
    PartitionData           = np.append(len(_SMat_EdgeData.data), Len_Indices)
    PartitionData           = np.append(PartitionData, Len_Indptr)
    PartitionData           = np.append(PartitionData, Partition_Indices)
    PartitionData           = np.append(PartitionData, Partition_Indptr)
    PartitionData           = PartitionData.astype(Dtype_All[1])
    PartitionData.tofile(_File_PartitionData)
    _File_PartitionData.close()

  # _Array_VertexOut[np.where(_Array_VertexOut == 0)] = 1
  # _Array_VertexIn[np.where(_Array_VertexIn == 0)] = 1

  _File_PartitionData = open(Str_DestDataPath + '/subdata/' + 'vertexout', 'w')
  _Array_VertexOut.tofile(_File_PartitionData)
  _File_PartitionData.close()
  _File_PartitionData = open(Str_DestDataPath + '/subdata/' + 'vertexin', 'w')
  _Array_VertexIn.tofile(_File_PartitionData)
  _File_PartitionData.close()
  return Str_DestDataPath + '/subdata/', Int_NewVertexNum, Int_PartitionNum, Int_VertexPerPartition

def intial_vertex(GraphInfo, Dtype_All, Str_Policy='ones'):
  if Str_Policy == 'ones':
    return np.ones(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
  elif Str_Policy == 'zeros':
    return np.zeros(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
  elif Str_Policy == 'random':
    temp = np.random.random(GraphInfo['VertexNum'])
    temp = temp.astype(Dtype_All['VertexData'])
    return temp
  elif Str_Policy == 'pagerank':
    temp = np.zeros(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
    temp = temp + 1.0 / GraphInfo['VertexNum']
    temp = temp.astype(Dtype_All['VertexData'])
    return temp
  else:
    return np.ones(GraphInfo['VertexNum'], dtype=Dtype_All[0])


def load_edgedata(PartitionID, GraphInfo, Dtype_All):
  _file = open(GraphInfo['DataPath'] + str(PartitionID) + '.edge', 'r')
  temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
  data = np.ones(temp[0], dtype=Dtype_All['EdgeData'])
  indices = temp[3:3 + int(temp[1])]
  indptr = temp[3 + int(temp[1]):3 + int(temp[1]) + int(temp[2])]
  mat_data = sparse.csr_matrix((data, indices, indptr), shape=(GraphInfo['VertexPerPartition'], GraphInfo['VertexNum']))
  _file.close()
  return mat_data


def load_vertexin(GraphInfo, Dtype_All):
  _file = open(GraphInfo['DataPath'] + 'vertexin', 'r')
  temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
  _file.close()
  return temp


def load_vertexout(GraphInfo, Dtype_All):
  _file = open(GraphInfo['DataPath'] + 'vertexout', 'r')
  temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
  _file.close()
  return temp


def calc_pagerank(PartitionID, DataInfo, GraphInfo, Dtype_All):
  UpdatedVertex = DataInfo['EdgeData'][PartitionID].dot(DataInfo['VertexData'] / DataInfo['VertexOut']) * 0.85 + 1.0 / GraphInfo['VertexNum']
  UpdatedVertex = UpdatedVertex.astype(Dtype_All['VertexData'])
  return UpdatedVertex


class BroadThread(threading.Thread):
  __MPIInfo     = {}
  __DataInfo    = None
  __GraphInfo   = {}
  __Dtype_All   = {}
  __ControlInfo = None
  __stop        = None

  def __init__(self, MPIInfo, DataInfo, ControlInfo, GraphInfo, Dtype_All):
    threading.Thread.__init__(self)
    self.__MPIInfo     = MPIInfo
    self.__DataInfo    = DataInfo
    self.__ControlInfo = ControlInfo
    self.__GraphInfo   = GraphInfo
    self.__Dtype_All   = Dtype_All
    self.__stop        = threading.Event()

  def stop(self):
    self.__stop.set()

  def run(self):
    while True:
      if self.__MPIInfo['MPI_Rank'] == 0:
        Str_UpdatedVertex = None
        Str_UpdatedVertex = QueueUpdatedVertex.get()
      else:
        Str_UpdatedVertex = None
      Str_UpdatedVertex = self.__MPIInfo['MPI_Comm'].bcast(Str_UpdatedVertex, root=0)
      if len(Str_UpdatedVertex) == 4 and Str_UpdatedVertex == 'exit':
        break
      Str_UpdatedVertex = snappy.decompress(Str_UpdatedVertex)
      updated_vertex = np.fromstring(Str_UpdatedVertex, dtype=self.__Dtype_All['VertexData'])
      if int(updated_vertex[-1]) in self.__ControlInfo['PartitionInfo'][self.__MPIInfo['MPI_Rank']]:
        pass
      else:
        start_id = int(updated_vertex[-1]) * self.__GraphInfo['VertexPerPartition']
        end_id = (int(updated_vertex[-1]) + 1) * self.__GraphInfo['VertexPerPartition']
        self.__DataInfo['VertexData'][start_id:end_id] = updated_vertex[0:-1] + self.__DataInfo['VertexData'][start_id:end_id]
      self.__ControlInfo['IterationReport'][int(updated_vertex[-1])] = self.__ControlInfo['IterationReport'][int(updated_vertex[-1])] + 1


class UpdateThread(threading.Thread):
  __MPIInfo   = {}
  __GraphInfo = {}
  __IP        = '127.0.0.1'
  __Port      = 17070
  __Dtype_All = {}
  __stop      = None

  def stop(self, Rank):
    if (Rank == 0):
      self.__stop.set()
      context_ = zmq.Context()
      socket_ = context_.socket(zmq.REQ)
      socket_.connect("tcp://%s:%s" % (self.__IP, self.__Port))
      socket_.send("exit")
      socket_.recv()
    else:
      self.__stop.set()
      QueueUpdatedVertex.put('exit')

  def __init__(self, IP, Port, MPIInfo, GraphInfo, Dtype_All):
    threading.Thread.__init__(self)
    self.__IP        = IP
    self.__Port      = Port
    self.__MPIInfo   = MPIInfo
    self.__GraphInfo = GraphInfo
    self.__Dtype_All = Dtype_All
    self.__stop      = threading.Event()

  def run(self):
    if self.__MPIInfo['MPI_Rank'] == 0:
      context = zmq.Context()
      socket = context.socket(zmq.REP)
      print(self.__IP, self.__Port)
      socket.bind("tcp://*:%s" % self.__Port)
      while True:
        string_receive = socket.recv()
        QueueUpdatedVertex.put(string_receive)
        socket.send("ACK")
        #                 print len(string_receive)
        if len(string_receive) == 4 and string_receive == 'exit':
          break
    else:
      while True:
        Str_UpdatedVertex = QueueUpdatedVertex.get()
        if len(Str_UpdatedVertex) == 4 and Str_UpdatedVertex == 'exit':
          break
        if self.__stop.is_set():
          break
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://%s:%s" % (self.__IP, self.__Port))
        socket.send(Str_UpdatedVertex)
        socket.recv()


class CalcThread(threading.Thread):
  __GraphInfo        = {}
  __Dtype_All        = {}
  __ControlInfo      = None
  __DataInfo         = None
  __PendingTaskQueue = None
  __RunningFlag      = False
  __stop             = threading.Event()

  def stop(self):
    self.__stop.set()

  def __init__(self, DataInfo, GraphInfo, ControlInfo, Dtype_All):
    threading.Thread.__init__(self)
    self.__DataInfo         = DataInfo
    self.__GraphInfo        = GraphInfo
    self.__ControlInfo      = ControlInfo
    self.__Dtype_All        = Dtype_All
    self.__PendingTaskQueue = Queue.Queue()
    __RunningFlag           = False

  def put_task(self, PartitionID):
    self.__PendingTaskQueue.put(PartitionID)

  def is_free(self):
    if self.__RunningFlag == False and self.__PendingTaskQueue.empty():
      return True

  def run(self):
    while True:
      self.__RunningFlag = False

      try:
        PartitionID_ = self.__PendingTaskQueue.get(timeout=0.05)
      except:
        if self.__stop.is_set():
          break
        else:
          continue
      self.__RunningFlag = True
      UpdatedVertex      = self.__ControlInfo['CalcFunc'](PartitionID_, self.__DataInfo, self.__GraphInfo, self.__Dtype_All)
      start_id           = PartitionID_ * self.__GraphInfo['VertexPerPartition']
      end_id             = (PartitionID_ + 1) * self.__GraphInfo['VertexPerPartition']
      UpdatedVertex      = UpdatedVertex - self.__DataInfo['VertexData'][start_id:end_id]
      UpdatedVertex[np.where(abs(UpdatedVertex) <= self.__ControlInfo['FilterThreshold'])] = 0
      UpdatedVertex      = UpdatedVertex.astype(self.__Dtype_All['VertexData'])
      #Update local data
      self.__DataInfo['VertexData'][start_id:end_id] = self.__DataInfo['VertexData'][start_id:end_id] + UpdatedVertex
      Tmp_UpdatedData = np.append(UpdatedVertex, PartitionID_)
      Tmp_UpdatedData = Tmp_UpdatedData.astype(self.__Dtype_All['VertexData'])
      Str_UpdatedData = Tmp_UpdatedData.tostring()
      Str_UpdatedData = snappy.compress(Str_UpdatedData)
      QueueUpdatedVertex.put(Str_UpdatedData)


class satgraph():
  __Dtype_All   = {}
  __GraphInfo   = {}
  __MPIInfo     = {}
  __ControlInfo = {}
  __DataInfo    = {}
  __Port        = 17070
  __IP          = '127.0.0.1'
  __ThreadNum   = 1

  def __init__(self):
    self.__Dtype_All['VertexData']         = np.int32
    self.__Dtype_All['VertexEdgeInfo']     = np.int32
    self.__Dtype_All['EdgeData']           = np.int32
    self.__DataPath                        = './subdata/'
    self.__VertexNum                       = 0
    self.__PartitionNum                    = 0
    self.__VertexPerPartition              = 0
    self.__GraphInfo['DataPath']           = self.__DataPath
    self.__GraphInfo['VertexNum']          = self.__VertexNum
    self.__GraphInfo['PartitionNum']       = self.__PartitionNum
    self.__GraphInfo['VertexPerPartition'] = self.__VertexPerPartition
    self.__ControlInfo['PartitionInfo']    = {}
    self.__ControlInfo['IterationNum']     = 0
    self.__ControlInfo['IterationReport']  = None
    self.__ControlInfo['MaxIteration']     = 10
    self.__ControlInfo['StaleNum']         = 0
    self.__ControlInfo['FilterThreshold']  = 0
    self.__ControlInfo['CalcFunc']         = None
    self.__DataInfo['EdgeData']            = {}
    self.__DataInfo['VertexOut']           = None
    self.__DataInfo['VertexIn']            = None
    self.__DataInfo['VertexData']          = None

  def set_FilterThreshold(self, FilterThreshold):
    self.__ControlInfo['FilterThreshold'] = FilterThreshold

  def set_StaleNum(self, StaleNum):
    self.__ControlInfo['StaleNum'] = StaleNum

  def set_CalcFunc(self, CalcFunc):
    self.__ControlInfo['CalcFunc'] = CalcFunc

  def set_ThreadNum(self, ThreadNum):
    self.__ThreadNum = ThreadNum

  def set_MaxIteration(self, MaxIteration):
    self.__ControlInfo['MaxIteration'] = MaxIteration

  def set_port(self, Port):
    self.__Port = Port

  def set_IP(self, IP):
    self.__IP = IP

  def set_GraphInfo(self, GraphInfo):
    self.__DataPath                        = GraphInfo[0]
    self.__VertexNum                       = GraphInfo[1]
    self.__PartitionNum                    = GraphInfo[2]
    self.__VertexPerPartition              = GraphInfo[3]
    self.__GraphInfo['DataPath']           = self.__DataPath
    self.__GraphInfo['VertexNum']          = self.__VertexNum
    self.__GraphInfo['PartitionNum']       = self.__PartitionNum
    self.__GraphInfo['VertexPerPartition'] = self.__VertexPerPartition
    self.__ControlInfo['IterationReport']  = np.zeros(self.__GraphInfo['PartitionNum'])

  def set_Dtype_All(self, Dtype_All):
    self.__Dtype_All['VertexData'] = Dtype_All[0]
    self.__Dtype_All['VertexEdgeInfo'] = Dtype_All[1]
    self.__Dtype_All['EdgeData'] = Dtype_All[2]

  def set_Dtype_VertexData(self, Dtype_VertexData):
    self.__Dtype_All['VertexData'] = Dtype_VertexData

  def set_Dtype_VertexEdgeInfo(self, Dtype_VertexEdgeInfo):
    self.__Dtype_All['VertexEdgeInfo'] = Dtype_VertexEdgeInfo

  def set_Dtype_EdgeData(self, Dtype_EdgeData):
    self.__Dtype_All['EdgeData'] = Dtype_VertexEdgeInfo

  @property
  def ControlInfo(self):
    return self.__ControlInfo

  @property
  def CalcFunc(self):
    return self.__CalcFunc

  @property
  def Port(self):
    return self.__Port

  @property
  def ThreadNum(self):
    return self.__ThreadNum

  @property
  def IP(self):
    return self.__IP

  @property
  def Dtype_All(self):
    return self.__Dtype_All

  @property
  def GraphInfo(self):
    return self.__GraphInfo

  @property
  def DataInfo(self):
    return self.__DataInfo

  def __wait_for_threadslot(self, TaskThreadPool_):
    while True:
      sleep(0.01)
      for i in range(self.__ThreadNum):
        if TaskThreadPool_[i].is_free():
          return i

  def __sync(self):
    while True:
      if (self.__ControlInfo['IterationNum'] \
              - self.__ControlInfo['IterationReport'].min()) \
              <= self.__ControlInfo['StaleNum']:
        break
      sleep(0.01)

  def __MPI_Initial(self):
    self.__MPIInfo['MPI_Comm'] = MPI.COMM_WORLD
    self.__MPIInfo['MPI_Size'] = self.__MPIInfo['MPI_Comm'].Get_size()
    self.__MPIInfo['MPI_Rank'] = self.__MPIInfo['MPI_Comm'].Get_rank()

  def run(self, Str_InitialVertex='zero'):
    self.__MPI_Initial()
    PartitionPerNode_ = int(math.floor(self.__GraphInfo['PartitionNum'] * 1.0 / self.__MPIInfo['MPI_Size']))
    for i in range(self.__MPIInfo['MPI_Size']):
      if i == self.__MPIInfo['MPI_Size'] - 1:
        self.__ControlInfo['PartitionInfo'][i] = range(i * PartitionPerNode_, self.__GraphInfo['PartitionNum'])
      else:
        self.__ControlInfo['PartitionInfo'][i] = range(i * PartitionPerNode_, (i + 1) * PartitionPerNode_)

    #load data to cache
    for i in self.__ControlInfo['PartitionInfo'][self.__MPIInfo['MPI_Rank']]:
      self.__DataInfo['EdgeData'][i] = load_edgedata(i, self.__GraphInfo, self.__Dtype_All)
    self.__DataInfo['VertexOut'] = load_vertexout(self.__GraphInfo, self.__Dtype_All)
    #Initial the vertex data
    self.__DataInfo['VertexData'] = intial_vertex(self.__GraphInfo, self.__Dtype_All, Str_InitialVertex)
    #Communication Thread
    UpdateVertexThread = UpdateThread(self.__IP, self.__Port, self.__MPIInfo, self.__GraphInfo, self.__Dtype_All)
    UpdateVertexThread.start()
    #BroadVertexThread Thread
    BroadVertexThread = BroadThread(self.__MPIInfo, self.__DataInfo, self.__ControlInfo, self.__GraphInfo, self.__Dtype_All)
    BroadVertexThread.start()

    AllTaskQueue = Queue.Queue()
    for i in range(self.__ControlInfo['MaxIteration']):
      for j in self.__ControlInfo['PartitionInfo'][self.__MPIInfo['MPI_Rank']]:
        AllTaskQueue.put(j)
    TaskThreadPool = []
    TaskTotalNum = 0

    for i in range(self.__ThreadNum):
      new_thead = CalcThread(self.__DataInfo, self.__GraphInfo, self.__ControlInfo, self.__Dtype_All)
      TaskThreadPool.append(new_thead)
      new_thead.start()

    if self.__MPIInfo['MPI_Rank'] == 0:
      Old_Vertex_ = self.__DataInfo['VertexData'].copy()

    while not AllTaskQueue.empty():
      free_threadid = self.__wait_for_threadslot(TaskThreadPool)
      new_partion = AllTaskQueue.get()
      CurrentIterationNum = TaskTotalNum / len(self.__ControlInfo['PartitionInfo'][self.__MPIInfo['MPI_Rank']])
      NewIteration = False
      if self.__ControlInfo['IterationNum'] != CurrentIterationNum:
        self.__ControlInfo['IterationNum'] = CurrentIterationNum
        NewIteration = True
      TaskTotalNum = TaskTotalNum + 1
      self.__sync()
      TaskThreadPool[free_threadid].put_task(new_partion)
      if NewIteration:
        if self.__MPIInfo['MPI_Rank'] == 0:
          print CurrentIterationNum, '->', 10000 * LA.norm(self.__DataInfo['VertexData'] - Old_Vertex_)
          Old_Vertex_ = self.__DataInfo['VertexData'].copy()

    for i in range(self.__ThreadNum):
      TaskThreadPool[i].stop()
    if (self.__MPIInfo['MPI_Rank'] != 0):
      UpdateVertexThread.stop(-1)
    else:
      sleep(0.1)
      UpdateVertexThread.stop(0)
    BroadVertexThread.stop()
    BroadVertexThread.join()
    print "BroadVertexThread->", self.__MPIInfo['MPI_Rank']
    UpdateVertexThread.join()
    print "UpdateVertexThread->", self.__MPIInfo['MPI_Rank']


if __name__ == '__main__':
  Dtype_VertexData     = np.float32
  Dtype_VertexEdgeInfo = np.int32
  Dtype_EdgeData       = np.bool
  Dtype_All            = (Dtype_VertexData, Dtype_VertexEdgeInfo, Dtype_EdgeData)

  # DataPath             = './subdata/'
  # VertexNum            = 81310
  # PartitionNum         = 10
  # VertexPerPartition   = 8131
  # GraphInfo            = (DataPath, VertexNum, PartitionNum, VertexPerPartition)
  # test_graph           = satgraph()

  # test_graph.set_Dtype_All(Dtype_All)
  # test_graph.set_GraphInfo(GraphInfo)
  # test_graph.set_IP('localhost')
  # test_graph.set_port(18085)
  # test_graph.set_ThreadNum(2)
  # test_graph.set_MaxIteration(50)
  # test_graph.set_StaleNum(0)
  # test_graph.set_FilterThreshold(0)
  # test_graph.set_CalcFunc(calc_pagerank)

  #a = preprocess_graph('./twitter.txt', './twitter2.txt', ' ');
  GraphInfo = graph_to_matrix('/data/4/eu-2015-t', './', 1070557254, 4000, 9179, Dtype_All);
  #GraphInfo = graph_to_matrix('/data/4/eu-2015-t', './', 1070557254, 4000, 1, Dtype_All);
  # GraphInfo = graph_to_matrix('/data/3/eu-2015.txt', './', 1070557254, 2000, Dtype_All);
  print GraphInfo;
  #81310, 8131, 10
  # test_graph.run('pagerank')
  #os._exit(0);
  #print PartitionInfo[MPI_Rank];
