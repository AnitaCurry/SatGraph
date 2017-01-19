'''
Created on 14 Apr 2016

@author: sunshine
'''
import os
import time
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI
import threading
import Queue
import zmq
import snappy
from numpy import linalg as LA

QueueUpdatedVertex = Queue.Queue()
# BSP = True
BSP = False

def intial_vertex(GraphInfo,
                  Dtype_All,
                  Str_Policy='ones'):
    if Str_Policy == 'ones':
        return np.ones(GraphInfo['VertexNum'],
                       dtype=Dtype_All['VertexData'])
    elif Str_Policy == 'zeros':
        return np.zeros(GraphInfo['VertexNum'],
                        dtype=Dtype_All['VertexData'])
    elif Str_Policy == 'random':
        temp = np.random.random(GraphInfo['VertexNum'])
        temp = temp.astype(Dtype_All['VertexData'])
        return temp
    elif Str_Policy == 'pagerank':
        temp = np.zeros(GraphInfo['VertexNum'],
                        dtype=Dtype_All['VertexData'])
        temp = temp + 1.0 / GraphInfo['VertexNum']
        temp = temp.astype(Dtype_All['VertexData'])
        return temp
    else:
        return np.ones(GraphInfo['VertexNum'],
                       dtype=Dtype_All[0])

def load_edgedata(PartitionID,
                  GraphInfo,
                  Dtype_All):
    _file = open(GraphInfo['DataPath'] +
                 str(PartitionID) +
                 '.edge',
                 'r')
    temp = np.fromfile(_file,
                       dtype=Dtype_All['VertexEdgeInfo'])
    data = np.ones(temp[0],
                   dtype=Dtype_All['EdgeData'])
    indices = temp[3:3 + int(temp[1])]
    indptr = temp[3 + int(temp[1]):3 + int(temp[1]) + int(temp[2])]

    encoded_data = (data, indices, indptr)
    encoded_shape = (GraphInfo['VertexPerPartition'], GraphInfo['VertexNum'])
    mat_data = sparse.csr_matrix(encoded_data, shape=encoded_shape)
    _file.close()
    return mat_data

def load_vertexin(GraphInfo,
                  Dtype_All):
    _file = open(GraphInfo['DataPath'] + 'vertexin', 'r')
    temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
    _file.close()
    return temp

def load_vertexout(GraphInfo,
                   Dtype_All):
    _file = open(GraphInfo['DataPath'] + 'vertexout', 'r')
    temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
    _file.close()
    return temp

def csr_row_set_nz_to_zero(csr, row):
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = 0

def csr_rows_set_nz_to_zero(csr, rows):
    for row in rows:
        csr_row_set_nz_to_zero(csr, row)
    # csr.eliminate_zeros()

def calc_pagerank(PartitionID,
                  IterationNum,
                  DataInfo,
                  GraphInfo,
                  Dtype_All):
    start_id = PartitionID * GraphInfo['VertexPerPartition']
    end_id = (PartitionID + 1) * GraphInfo['VertexPerPartition']
    GraphMatrix = load_edgedata(PartitionID, GraphInfo, Dtype_All)
    VertexVersion = DataInfo['VertexVersion'][start_id:end_id]

    ActiveRow = np.where(VertexVersion >= (IterationNum-2))[0]
    DeactiveRow = np.where(VertexVersion <  (IterationNum-2))[0]

 #   if MPI.COMM_WORLD.Get_rank() == 0:
 #       print len(ActiveRow)*1.0/GraphMatrix.shape[0]

    if len(ActiveRow)*1.0/GraphMatrix.shape[0] <= 0.01:
        UpdatedVertex = DataInfo['VertexData'][start_id:end_id].copy()
        if len(ActiveRow) == 0:
            UpdatedVertex = DataInfo['VertexData'][start_id:end_id].copy()
            return UpdatedVertex
#########################################################################
#        csr_rows_set_nz_to_zero(GraphMatrix, DeactiveRow)
#        NormlizedVertex = DataInfo['VertexData'] / DataInfo['VertexOut']
#        UpdatedVertex = GraphMatrix.dot(NormlizedVertex) * 0.85
#        UpdatedVertex = UpdatedVertex + 1.0 / GraphInfo['VertexNum']
#        UpdatedVertex[DeactiveRow] = \
#            DataInfo['VertexData'][start_id:end_id][DeactiveRow].copy()
#########################################################################
        NormlizedVertex = DataInfo['VertexData'] / DataInfo['VertexOut']
        for i in ActiveRow:   
            UpdatedVertex[i] = GraphMatrix[i].dot(NormlizedVertex) * 0.85
#        GraphMatrix = GraphMatrix[ActiveRow]
#        UpdatedVertex[ActiveRow] = \
#            GraphMatrix.dot(NormlizedVertex) * 0.85
        UpdatedVertex[ActiveRow] = \
            UpdatedVertex[ActiveRow] + 1.0 / GraphInfo['VertexNum']
        UpdatedVertex = UpdatedVertex.astype(Dtype_All['VertexData'])
        return UpdatedVertex
    else:
        NormlizedVertex = DataInfo['VertexData'] / DataInfo['VertexOut']
        UpdatedVertex = GraphMatrix.dot(NormlizedVertex) * 0.85
        UpdatedVertex = UpdatedVertex + 1.0 / GraphInfo['VertexNum']
        UpdatedVertex = UpdatedVertex.astype(Dtype_All['VertexData'])
        return UpdatedVertex

class BroadThread(threading.Thread):
    __MPIInfo = {}
    __DataInfo = None
    __GraphInfo = {}
    __Dtype_All = {}
    __ControlInfo = None
    __stop = None

    def __init__(self,
                 MPIInfo,
                 DataInfo,
                 ControlInfo,
                 GraphInfo,
                 Dtype_All):
        threading.Thread.__init__(self)
        self.__MPIInfo = MPIInfo
        self.__DataInfo = DataInfo
        self.__ControlInfo = ControlInfo
        self.__GraphInfo = GraphInfo
        self.__Dtype_All = Dtype_All
        self.__stop = threading.Event()

    def stop(self):
        self.__stop.set()

    def broadcast(self):
        if self.__MPIInfo['MPI_Rank'] == 0:
            Str_UpdatedVertex = None
            Str_UpdatedVertex = QueueUpdatedVertex.get()
        else:
            Str_UpdatedVertex = None
        return self.__MPIInfo['MPI_Comm'].bcast(Str_UpdatedVertex, root=0)

    def update_BSP(self, updated_vertex, start_id, end_id):
        new_vertex = updated_vertex[0:-1] + \
            self.__DataInfo['VertexData'][start_id:end_id]
        self.__DataInfo['VertexDataNew'][start_id:end_id] = new_vertex
        # update vertex data
        i = int(updated_vertex[-1])
        self.__ControlInfo['IterationReport'][i] = \
            self.__ControlInfo['IterationReport'][i] + 1
        while True:
            if self.__ControlInfo['IterationNum'] == \
                    self.__ControlInfo['IterationReport'].min():
                break
            else:
                time.sleep(0.1)
        # update vertex version number
        version_num = self.__ControlInfo['IterationReport'][i]
        non_zero_id = np.where(updated_vertex[0:-1]!=0)[0]
        non_zero_id = non_zero_id + start_id
        self.__DataInfo['VertexVersion'][non_zero_id] = version_num

    def update_SSP(self, updated_vertex, start_id, end_id):
        new_vertex = updated_vertex[0:-1] + \
            self.__DataInfo['VertexData'][start_id:end_id]
        self.__DataInfo['VertexData'][start_id:end_id] = new_vertex
        # update vertex data
        i = int(updated_vertex[-1])
        self.__ControlInfo['IterationReport'][i] = \
            self.__ControlInfo['IterationReport'][i] + 1
        # update vertex version number
        version_num = self.__ControlInfo['IterationReport'][i]
        non_zero_id = np.where(updated_vertex[0:-1]!=0)[0]
        non_zero_id = non_zero_id + start_id
        self.__DataInfo['VertexVersion'][non_zero_id] = version_num

    def broadcast_process(self):
        Str_UpdatedVertex = self.broadcast()
        if len(Str_UpdatedVertex) == 4 and Str_UpdatedVertex == 'exit':
            return -1
        Str_UpdatedVertex = snappy.decompress(Str_UpdatedVertex)
        updated_vertex = np.fromstring(Str_UpdatedVertex,
                                       dtype=self.__Dtype_All['VertexData'])
        start_id = int(updated_vertex[-1]) * \
            self.__GraphInfo['VertexPerPartition']
        end_id = (int(updated_vertex[-1]) + 1) * \
            self.__GraphInfo['VertexPerPartition']

        if not BSP:
            self.update_SSP(updated_vertex, start_id, end_id)
        else:
            self.update_BSP(updated_vertex, start_id, end_id)
        return 1

    def run(self):
        while True:
            if self.broadcast_process() == -1:
                break

class UpdateThread(threading.Thread):
    __MPIInfo = {}
    __GraphInfo = {}
    __IP = '127.0.0.1'
    __UpdatePort = 17070
    __Dtype_All = {}
    __stop = None

    def stop(self, Rank):
        if (Rank == 0):
            self.__stop.set()
            context_ = zmq.Context()
            socket_ = context_.socket(zmq.REQ)
            socket_.connect("tcp://%s:%s" % (self.__IP, self.__UpdatePort))
            socket_.send("exit")
            socket_.recv()
        else:
            self.__stop.set()
            QueueUpdatedVertex.put('exit')

    def __init__(self,
                 IP,
                 Port,
                 MPIInfo,
                 GraphInfo,
                 Dtype_All):
        threading.Thread.__init__(self)
        self.__IP = IP
        self.__UpdatePort = Port
        self.__MPIInfo = MPIInfo
        self.__GraphInfo = GraphInfo
        self.__Dtype_All = Dtype_All
        self.__stop = threading.Event()

    def run(self):
        if self.__MPIInfo['MPI_Rank'] == 0:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            print(self.__IP, self.__UpdatePort)
            socket.bind("tcp://*:%s" % self.__UpdatePort)
            while True:
                string_receive = socket.recv()
                QueueUpdatedVertex.put(string_receive)
                socket.send("ACK")
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
                socket.connect("tcp://%s:%s" % (self.__IP, self.__UpdatePort))
                socket.send(Str_UpdatedVertex)
                socket.recv()

class CalcThread(threading.Thread):
    __GraphInfo = {}
    __Dtype_All = {}
    __ControlInfo = None
    __DataInfo = None
    __IP = None
    __Port = None
    __stop = threading.Event()

    def stop(self):
        self.__stop.set()

    def __init__(self,
                 DataInfo,
                 GraphInfo,
                 ControlInfo,
                 IP,
                 Port,
                 Dtype_All):
        threading.Thread.__init__(self)
        self.__DataInfo = DataInfo
        self.__GraphInfo = GraphInfo
        self.__ControlInfo = ControlInfo
        self.__Dtype_All = Dtype_All
        self.__IP = IP
        self.__Port = Port

    def sync(self):
        if self.__stop.is_set():
            return -1
        if BSP:
            while True:
                if self.__ControlInfo['IterationNum'] == \
                        self.__ControlInfo['IterationReport'].min():
                    break
                else:
                    time.sleep(0.1)
        return 1

    def run(self):
        while True:
            if self.sync() == -1:
                break
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://%s:%s" % (self.__IP, self.__Port))
            TaskRequest = '1 ' + str(MPI.COMM_WORLD.Get_rank())
            socket.send(TaskRequest)
            message = socket.recv()
            if message == '-1':
                time.sleep(0.5)
                continue

            i = int(message)
            UpdatedVertex = \
                self.__ControlInfo['CalcFunc'](i,
                                               self.__ControlInfo['IterationNum'],
                                               self.__DataInfo,
                                               self.__GraphInfo,
                                               self.__Dtype_All)
            start_id = i * \
                self.__GraphInfo['VertexPerPartition']
            end_id = (i + 1) * \
                self.__GraphInfo['VertexPerPartition']
            UpdatedVertex = UpdatedVertex - \
                self.__DataInfo['VertexData'][start_id:end_id]

            filterd_id = np.where(abs(UpdatedVertex) <=
                                  self.__ControlInfo['FilterThreshold'])
            UpdatedVertex[filterd_id] = 0

            UpdatedVertex = \
                UpdatedVertex.astype(self.__Dtype_All['VertexData'])
            Tmp_UpdatedData = np.append(UpdatedVertex, i)
            Tmp_UpdatedData = \
                Tmp_UpdatedData.astype(self.__Dtype_All['VertexData'])
            Str_UpdatedData = Tmp_UpdatedData.tostring()
            Str_UpdatedData = snappy.compress(Str_UpdatedData)
            QueueUpdatedVertex.put(Str_UpdatedData)

class SchedulerThread(threading.Thread):
    __MPIInfo = {}
    __GraphInfo = {}
    __IP = '127.0.0.1'
    __TaskqPort = 17071
    __Dtype_All = {}
    __stop = None

    def stop(self, Rank):
        self.__stop.set()
        context_ = zmq.Context()
        socket_ = context_.socket(zmq.REQ)
        socket_.connect("tcp://%s:%s" % (self.__IP, self.__TaskqPort))
        socket_.send("-1 -1")
        socket_.recv()

    def __init__(self,
                 IP,
                 Port,
                 MPIInfo,
                 GraphInfo,
                 ControlInfo,
                 Dtype_All):
        threading.Thread.__init__(self)
        self.__IP = IP
        self.__TaskqPort = Port
        self.__MPIInfo = MPIInfo
        self.__GraphInfo = GraphInfo
        self.__ControlInfo = ControlInfo
        self.__Dtype_All = Dtype_All
        self.__stop = threading.Event()

    def assign_task(self, AllTask, AllProgress, socket):
        if AllProgress.min() >= self.__ControlInfo['MaxIteration']:
            socket.send("-1")
        elif AllTask.min() >= self.__ControlInfo['MaxIteration']:
            socket.send("-1")
        elif AllProgress.max() - AllProgress.min() <= \
                self.__ControlInfo['StaleNum']:
            candicate_partition = np.where(
                AllTask - AllProgress == 0)[0]
            candicate_status = AllTask[candicate_partition]
            if len(candicate_partition) == 0:
                socket.send("-1")
            else:
                target_partition = candicate_partition[
                    candicate_status.argmin()]
                # print target_partition,' to ', data;
                AllTask[target_partition] = AllTask[
                    target_partition] + 1
                socket.send(str(target_partition))
        else:
            socket.send("-1")

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        print(self.__IP, self.__TaskqPort)
        socket.bind("tcp://*:%s" % self.__TaskqPort)

        AllTask = np.zeros(self.__GraphInfo['PartitionNum'], dtype=np.int32)
        AllProgress = self.__ControlInfo['IterationReport']
        while True:
            string_receive = socket.recv()
            command, data = string_receive.split()
            if command == '-1':  # exit
                socket.send("-1")
                break
            elif command == '1':  # get task
                self.assign_task(AllTask, AllProgress, socket)
            else:
                socket.send("-1")

class satgraph():
    __Dtype_All = {}
    __GraphInfo = {}
    __MPIInfo = {}
    __ControlInfo = {}
    __DataInfo = {}
    __UpdatePort = 17070
    __TaskqPort = 17071
    __IP = '127.0.0.1'
    __ThreadNum = 1

    def __init__(self):
        self.__Dtype_All['VertexData'] = np.int32
        self.__Dtype_All['VertexEdgeInfo'] = np.int32
        self.__Dtype_All['EdgeData'] = np.int32
        self.__DataPath = './subdata/'
        self.__VertexNum = 0
        self.__PartitionNum = 0
        self.__VertexPerPartition = 0
        self.__GraphInfo['DataPath'] = self.__DataPath
        self.__GraphInfo['VertexNum'] = self.__VertexNum
        self.__GraphInfo['PartitionNum'] = self.__PartitionNum
        self.__GraphInfo['VertexPerPartition'] = self.__VertexPerPartition
        self.__ControlInfo['IterationNum'] = 0
        self.__ControlInfo['IterationReport'] = None
        self.__ControlInfo['MaxIteration'] = 10
        self.__ControlInfo['StaleNum'] = 0
        self.__ControlInfo['FilterThreshold'] = 0
        self.__ControlInfo['CalcFunc'] = None
        self.__DataInfo['EdgeData'] = {}
        self.__DataInfo['VertexOut'] = None
        self.__DataInfo['VertexIn'] = None
        self.__DataInfo['VertexData'] = None
        self.__DataInfo['VertexVersion'] = None
        if BSP:
            self.__DataInfo['VertexDataNew'] = None

    def set_FilterThreshold(self, FilterThreshold):
        self.__ControlInfo['FilterThreshold'] = FilterThreshold

    def set_StaleNum(self, StaleNum):
        self.__ControlInfo['StaleNum'] = StaleNum
        if BSP:
            self.__ControlInfo['StaleNum'] = 1

    def set_CalcFunc(self, CalcFunc):
        self.__ControlInfo['CalcFunc'] = CalcFunc

    def set_ThreadNum(self, ThreadNum):
        self.__ThreadNum = ThreadNum

    def set_MaxIteration(self, MaxIteration):
        self.__ControlInfo['MaxIteration'] = MaxIteration

    def set_port(self, Port1, Port2):
        self.__UpdatePort = Port1
        self.__TaskqPort = Port2

    def set_IP(self, IP):
        self.__IP = IP

    def set_GraphInfo(self, GraphInfo):
        self.__DataPath = GraphInfo[0]
        self.__VertexNum = GraphInfo[1]
        self.__PartitionNum = GraphInfo[2]
        self.__VertexPerPartition = GraphInfo[3]
        self.__GraphInfo['DataPath'] = self.__DataPath
        self.__GraphInfo['VertexNum'] = self.__VertexNum
        self.__GraphInfo['PartitionNum'] = self.__PartitionNum
        self.__GraphInfo['VertexPerPartition'] = self.__VertexPerPartition
        self.__ControlInfo['IterationReport'] = np.zeros(
            self.__GraphInfo['PartitionNum'], dtype=np.int32)
        self.__DataInfo['VertexVersion'] = np.zeros(
            self.__GraphInfo['VertexNum'], dtype=np.int32)

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

    def __MPI_Initial(self):
        self.__MPIInfo['MPI_Comm'] = MPI.COMM_WORLD
        self.__MPIInfo['MPI_Size'] = self.__MPIInfo['MPI_Comm'].Get_size()
        self.__MPIInfo['MPI_Rank'] = self.__MPIInfo['MPI_Comm'].Get_rank()

    def graph_process(self):
        time.sleep(0.1)
        CurrentIterationNum = self.__ControlInfo['IterationReport'].min()
        NewIteration = False
        if self.__ControlInfo['IterationNum'] != CurrentIterationNum:
            NewIteration = True
            if BSP:
                self.__DataInfo['VertexData'] = \
                    self.__DataInfo['VertexDataNew'].copy()
            self.__ControlInfo['IterationNum'] = CurrentIterationNum
        return NewIteration, CurrentIterationNum

    def create_threads(self):
        UpdateVertexThread = \
            UpdateThread(self.__IP, self.__UpdatePort,
                         self.__MPIInfo, self.__GraphInfo,
                         self.__Dtype_All)
        UpdateVertexThread.start()

        TaskSchedulerThread = None
        if self.__MPIInfo['MPI_Rank'] == 0:
            TaskSchedulerThread = \
                SchedulerThread(self.__IP, self.__TaskqPort,
                                self.__MPIInfo, self.__GraphInfo,
                                self.__ControlInfo, self.__Dtype_All)
            TaskSchedulerThread.start()

        BroadVertexThread = \
            BroadThread(self.__MPIInfo, self.__DataInfo,
                        self.__ControlInfo, self.__GraphInfo,
                        self.__Dtype_All)
        BroadVertexThread.start()

        TaskThreadPool = []
        for i in range(self.__ThreadNum):
            new_task_thead = \
                CalcThread(self.__DataInfo, self.__GraphInfo,
                           self.__ControlInfo, self.__IP,
                           self.__TaskqPort, self.__Dtype_All)
            TaskThreadPool.append(new_task_thead)
            new_task_thead.start()
        return UpdateVertexThread, TaskSchedulerThread, \
            BroadVertexThread, TaskThreadPool

    def destroy_threads(self,
                        UpdateVertexThread,
                        TaskSchedulerThread,
                        BroadVertexThread,
                        TaskThreadPool):
        for i in range(self.__ThreadNum):
            TaskThreadPool[i].stop()
        if (self.__MPIInfo['MPI_Rank'] != 0):
            UpdateVertexThread.stop(-1)
        else:
            TaskSchedulerThread.stop(0)
            time.sleep(1)
            UpdateVertexThread.stop(0)
        BroadVertexThread.stop()
        BroadVertexThread.join()
        UpdateVertexThread.join()
        if self.__MPIInfo['MPI_Rank'] == 0:
            TaskSchedulerThread.join()

    def run(self, InitialVertex='zero'):
        self.__MPI_Initial()
        self.__DataInfo['VertexOut'] = \
            load_vertexout(self.__GraphInfo, self.__Dtype_All)
        self.__DataInfo['VertexData'] = \
            intial_vertex(self.__GraphInfo, self.__Dtype_All, InitialVertex)
        if BSP:
            self.__DataInfo['VertexDataNew'] = \
                intial_vertex(self.__GraphInfo,
                              self.__Dtype_All, InitialVertex)

        UpdateVertexThread, TaskSchedulerThread, \
            BroadVertexThread, TaskThreadPool = self.create_threads()

        if self.__MPIInfo['MPI_Rank'] == 0:
            Old_Vertex_ = self.__DataInfo['VertexData'].copy()
            start_time = time.time()
            app_start_time = time.time()

        while True:
            NewIteration, CurrentIteration = self.graph_process()
            if NewIteration and self.__MPIInfo['MPI_Rank'] == 0:
                end_time = time.time()
                diff_vertex = 10000 * \
                    LA.norm(self.__DataInfo['VertexData'] - Old_Vertex_)
                print end_time - start_time, ' # Iter: ', \
                    CurrentIteration, '->', diff_vertex
                Old_Vertex_ = self.__DataInfo['VertexData'].copy()
                start_time = time.time()
            if CurrentIteration == self.__ControlInfo['MaxIteration']:
                break

        if self.__MPIInfo['MPI_Rank'] == 0:
            app_end_time = time.time()
            print 'Time Used: ', app_end_time - app_start_time

        self.destroy_threads(UpdateVertexThread, TaskSchedulerThread,
                             BroadVertexThread, TaskThreadPool)

if __name__ == '__main__':
    Dtype_VertexData = np.float32
    Dtype_VertexEdgeInfo = np.int32
    Dtype_EdgeData = np.bool
    Dtype_All = (Dtype_VertexData, Dtype_VertexEdgeInfo, Dtype_EdgeData)

    # DataPath = '/home/mapred/GraphData/wiki/subdata/'
    # VertexNum = 4206800
    # PartitionNum = 20
    #
    DataPath = '/home/mapred/GraphData/uk/subdata/'
    VertexNum = 787803000
    PartitionNum = 3000

    # DataPath = '/home/mapred/GraphData/twitter/subdata/'
    # VertexNum = 41652250
    # PartitionNum = 50

    GraphInfo = (DataPath, VertexNum, PartitionNum, VertexNum / PartitionNum)
    test_graph = satgraph()

    rank_0_host = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        rank_0_host = MPI.Get_processor_name()
    rank_0_host = MPI.COMM_WORLD.bcast(rank_0_host, root=0)

    test_graph.set_Dtype_All(Dtype_All)
    test_graph.set_GraphInfo(GraphInfo)
    test_graph.set_IP(rank_0_host)
    test_graph.set_port(18086, 18087)
    test_graph.set_ThreadNum(4)
    test_graph.set_MaxIteration(100)
    test_graph.set_StaleNum(1)
    # test_graph.set_FilterThreshold(0)
    test_graph.set_FilterThreshold(0.0000001)
    test_graph.set_CalcFunc(calc_pagerank)

    test_graph.run('pagerank')
    os._exit(0)
