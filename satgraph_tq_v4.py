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
from functools import partial
import ctypes

QueueUpdatedVertex = Queue.Queue()
#BSP = True
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
        temp += 1.0 / GraphInfo['VertexNum']
        temp = temp.astype(Dtype_All['VertexData'])
        return temp
    else:
        return np.ones(GraphInfo['VertexNum'],
                       dtype=Dtype_All[0])

def load_edgedata(PartitionID,
                  GraphInfo,
                  Dtype_All):
    edge_path = GraphInfo['DataPath'] + str(PartitionID) + '.edge'
    _file = open(edge_path, 'r')
    temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
    data = np.ones(temp[0], dtype=Dtype_All['EdgeData'])
    indices = temp[5:5 + int(temp[1])]
    indptr = temp[5 + int(temp[1]):5 + int(temp[1]) + int(temp[2])]
    start_id = int(temp[3])
    end_id = int(temp[4])

    encoded_data = (data, indices, indptr)
    encoded_shape = (end_id-start_id, GraphInfo['VertexNum'])
    mat_data = sparse.csr_matrix(encoded_data, shape=encoded_shape)
    _file.close()
    return mat_data, start_id, end_id

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
    # temp[np.where(temp==0)] = 1
    _file.close()
    return temp

def calc_pagerank(PartitionID,
                  IterationNum,
                  DataInfo,
                  GraphInfo,
                  Dtype_All):
    EdgeMatrix, start_id, end_id = \
        load_edgedata(PartitionID, GraphInfo, Dtype_All)
    VertexVersion = DataInfo['VertexVersion'][start_id:end_id]
    ActiveVertex = np.where(VertexVersion >= (IterationNum-3))[0]
    # DeactiveVertex = np.where(VertexVersion < (IterationNum-3))[0]

    UpdatedVertex = np.zeros(end_id-start_id, dtype=Dtype_All['VertexData'])
    if len(ActiveVertex) == 0:
        UpdatedVertex[:] = DataInfo['VertexData'][start_id:end_id][:]
        return UpdatedVertex, start_id, end_id

    if len(ActiveVertex) <= 3000:
        UpdatedVertex[:] = DataInfo['VertexData'][start_id:end_id][:]
        EdgeMatrix = EdgeMatrix[ActiveVertex]
        NormlizedVertex = DataInfo['VertexData'] / DataInfo['VertexOut']
        UpdatedVertex[ActiveVertex] = EdgeMatrix.dot(NormlizedVertex) * 0.85
        UpdatedVertex[ActiveVertex] += 1.0 / GraphInfo['VertexNum']
    else:
        NormlizedVertex = DataInfo['VertexData'] / DataInfo['VertexOut']
        UpdatedVertex = EdgeMatrix.dot(NormlizedVertex) * 0.85
        UpdatedVertex += 1.0 / GraphInfo['VertexNum']

    UpdatedVertex = UpdatedVertex.astype(Dtype_All['VertexData'])
    return UpdatedVertex, start_id, end_id

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
        new_vertex = updated_vertex[0:-5] + \
            self.__DataInfo['VertexData'][start_id:end_id]
        self.__DataInfo['VertexDataNew'][start_id:end_id] = new_vertex
        # update vertex data
        i = int(updated_vertex[-5])
        self.__ControlInfo['IterationReport'][i] += 1
        while 1:
            if self.__ControlInfo['IterationNum'] == \
                    self.__ControlInfo['IterationReport'].min():
                break
            else:
                time.sleep(0.1)
        # update vertex version number
        version_num = self.__ControlInfo['IterationReport'][i]
        non_zero_id = np.where(updated_vertex[0:-5]!=0)[0]
        non_zero_id += start_id
        self.__DataInfo['VertexVersion'][non_zero_id] = version_num

    def update_SSP(self, updated_vertex, start_id, end_id):
        self.__DataInfo['VertexData'][start_id:end_id] += updated_vertex[0:-5]
        # update vertex data
        i = int(updated_vertex[-5])
        self.__ControlInfo['IterationReport'][i] += 1
        # update vertex version number
        version_num = self.__ControlInfo['IterationReport'][i]
        non_zero_id = np.where(updated_vertex[0:-5]!=0)[0]
        non_zero_id += start_id
        self.__DataInfo['VertexVersion'][non_zero_id] = version_num

    def broadcast_process(self):
        Str_UpdatedVertex = self.broadcast()
        if len(Str_UpdatedVertex) == 4 and Str_UpdatedVertex == 'exit':
            return -1
        Str_UpdatedVertex = snappy.decompress(Str_UpdatedVertex)
        updated_vertex = np.fromstring(Str_UpdatedVertex,
                                       dtype=self.__Dtype_All['VertexData'])
        start_id = int(updated_vertex[-4])*1000000+int(updated_vertex[-3])
        end_id   = int(updated_vertex[-2])*1000000+int(updated_vertex[-1])

        if not BSP:
            self.update_SSP(updated_vertex, start_id, end_id)
        else:
            self.update_BSP(updated_vertex, start_id, end_id)
        return 1

    def run(self):
        while 1:
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
            while 1:
                string_receive = socket.recv()
                QueueUpdatedVertex.put(string_receive)
                socket.send("ACK")
                if len(string_receive) == 4 and string_receive == 'exit':
                    break
        else:
            while 1:
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
            while 1:
                if self.__ControlInfo['IterationNum'] == \
                        self.__ControlInfo['IterationReport'].min():
                    break
                else:
                    time.sleep(0.1)
        return 1

    def run(self):
        while 1:
            if self.sync() == -1:
                break
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://%s:%s" % (self.__IP, self.__Port))
            TaskRequest = '1 ' + str(MPI.COMM_WORLD.Get_rank())
            socket.send(TaskRequest)
            message = socket.recv()
            if message == '-1':
                time.sleep(0.1)
                continue

            i = int(message)
            UpdatedVertex, start_id, end_id = \
                self.__ControlInfo['CalcFunc'](i,
                                               self.__ControlInfo['IterationNum'],
                                               self.__DataInfo,
                                               self.__GraphInfo,
                                               self.__Dtype_All)
            UpdatedVertex -= self.__DataInfo['VertexData'][start_id:end_id]
            filterd_id = np.where(abs(UpdatedVertex) <=
                                  self.__ControlInfo['FilterThreshold'])
            UpdatedVertex[filterd_id] = 0
            UpdatedVertex = \
                UpdatedVertex.astype(self.__Dtype_All['VertexData'])
            UpdatedVertex = np.append(UpdatedVertex, i)
            UpdatedVertex = np.append(UpdatedVertex, int(start_id/1000000))
            UpdatedVertex = np.append(UpdatedVertex, start_id%1000000)
            UpdatedVertex = np.append(UpdatedVertex, int(end_id/1000000))
            UpdatedVertex = np.append(UpdatedVertex, end_id%1000000)
            UpdatedVertex = UpdatedVertex.astype(self.__Dtype_All['VertexData'])

            UpdatedVertex = UpdatedVertex.tostring()
            UpdatedVertex = snappy.compress(UpdatedVertex)
            QueueUpdatedVertex.put(UpdatedVertex)

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

    def assign_task(self, rank, LocalityInfo, AllTask, AllProgress, socket):
        if AllProgress.min() >= self.__ControlInfo['MaxIteration']:
            socket.send("-1")
        elif AllTask.min() >= self.__ControlInfo['MaxIteration']:
            socket.send("-1")
        elif AllProgress.max() - AllProgress.min() <= \
                self.__ControlInfo['StaleNum']:
            candicate_partition = np.where(AllTask - AllProgress == 0)[0]
            if len(candicate_partition) == 0:
                socket.send("-1")
            else:
                candicate_status = AllTask[candicate_partition]
                target_status = candicate_status.min()
                target_ids = np.where(candicate_status == target_status)[0]
                target_partition = candicate_partition[target_ids]
                target_locality = LocalityInfo[rank][target_partition]
                max_allocate = target_locality.argmax()
                target_partition = target_partition[max_allocate]

                AllTask[target_partition] += 1
                LocalityInfo[rank][target_partition] += 1
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
        LocalityInfo = {}
        for i in range(MPI.COMM_WORLD.Get_size()):
            LocalityInfo[i] = np.zeros(self.__GraphInfo['PartitionNum'],
                                       dtype=np.int32)

        while 1:
            string_receive = socket.recv()
            command, rank = string_receive.split()
            if command == '-1':  # exit
                socket.send("-1")
                break
            elif command == '1':  # get task
                rank = int(rank)
                self.assign_task(rank, LocalityInfo, AllTask, AllProgress, socket)
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
        self.__GraphInfo['DataPath'] = None
        self.__GraphInfo['VertexNum'] = None
        self.__GraphInfo['PartitionNum'] = None
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
        self.__GraphInfo['DataPath'] = GraphInfo[0]
        self.__GraphInfo['VertexNum'] = GraphInfo[1]
        self.__GraphInfo['PartitionNum'] = GraphInfo[2]
        self.__ControlInfo['IterationReport'] = np.zeros(
            self.__GraphInfo['PartitionNum'], dtype=np.int32)
        self.__DataInfo['VertexVersion'] = np.zeros(
            self.__GraphInfo['VertexNum'], dtype=np.int32)

    def set_Dtype_All(self, Dtype_All):
        self.__Dtype_All['VertexData'] = Dtype_All[0]
        self.__Dtype_All['VertexEdgeInfo'] = Dtype_All[1]
        self.__Dtype_All['EdgeData'] = Dtype_All[2]

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
                self.__DataInfo['VertexData'][:] = self.__DataInfo['VertexDataNew'][:]
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

        MPI.COMM_WORLD.Barrier()

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

        while 1:
            NewIteration, CurrentIteration = self.graph_process()
            if NewIteration and self.__MPIInfo['MPI_Rank'] == 0:
                end_time = time.time()
                diff_vertex = 10000 * \
                    LA.norm(self.__DataInfo['VertexData'] - Old_Vertex_)
                print end_time - start_time, ' # Iter: ', \
                    CurrentIteration, '->', diff_vertex
                Old_Vertex_[:] = self.__DataInfo['VertexData'][:]
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

    # DataPath = '/home/mapred/GraphData/wiki/edge/'
    # VertexNum = 4206800
    # PartitionNum = 21
    #
    DataPath = '/home/mapred/GraphData/uk/edge/'
    VertexNum = 787803000
    PartitionNum = 3170

    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(2)))


    # DataPath = '/home/mapred/GraphData/twitter/edge/'
    # VertexNum = 41652250
    # PartitionNum = 49

    GraphInfo = (DataPath, VertexNum, PartitionNum)
    test_graph = satgraph()

    rank_0_host = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        rank_0_host = MPI.Get_processor_name()
    rank_0_host = MPI.COMM_WORLD.bcast(rank_0_host, root=0)

    test_graph.set_Dtype_All(Dtype_All)
    test_graph.set_GraphInfo(GraphInfo)
    test_graph.set_IP(rank_0_host)
    test_graph.set_port(18086, 18087)
    test_graph.set_ThreadNum(6)
    test_graph.set_MaxIteration(50)
    test_graph.set_StaleNum(3)
    test_graph.set_FilterThreshold(10**(-7))
    # test_graph.set_FilterThreshold(1/VertexNum)
    test_graph.set_CalcFunc(calc_pagerank)

    MPI.COMM_WORLD.Barrier()

    test_graph.run('pagerank')
    os._exit(0)
