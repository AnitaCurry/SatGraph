'''
Created on 14 Apr 2016

@author: sunshine
'''
import os
import time
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
import pandas as pd

QueueUpdatedVertex = Queue.Queue()
#BSP = True
BSP = False

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
    mat_data = sparse.csr_matrix((data, indices, indptr), shape=(
        GraphInfo['VertexPerPartition'], GraphInfo['VertexNum']))
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

    GraphMatrix = load_edgedata(PartitionID, GraphInfo, Dtype_All)

    UpdatedVertex = GraphMatrix.dot(
        DataInfo['VertexData'] / DataInfo['VertexOut']) * 0.85 + 1.0 / GraphInfo['VertexNum']
    UpdatedVertex = UpdatedVertex.astype(Dtype_All['VertexData'])
    return UpdatedVertex


class BroadThread(threading.Thread):
    __MPIInfo = {}
    __DataInfo = None
    __GraphInfo = {}
    __Dtype_All = {}
    __ControlInfo = None
    __stop = None

    def __init__(self, MPIInfo, DataInfo, ControlInfo, GraphInfo, Dtype_All):
        threading.Thread.__init__(self)
        self.__MPIInfo = MPIInfo
        self.__DataInfo = DataInfo
        self.__ControlInfo = ControlInfo
        self.__GraphInfo = GraphInfo
        self.__Dtype_All = Dtype_All
        self.__stop = threading.Event()

    def stop(self):
        self.__stop.set()

    def run(self):
        while True:
            if self.__MPIInfo['MPI_Rank'] == 0:
                Str_UpdatedVertex = None
                Str_UpdatedVertex = QueueUpdatedVertex.get()
            else:
                Str_UpdatedVertex = None
            Str_UpdatedVertex = self.__MPIInfo[
                'MPI_Comm'].bcast(Str_UpdatedVertex, root=0)
            if len(Str_UpdatedVertex) == 4 and Str_UpdatedVertex == 'exit':
                break
            Str_UpdatedVertex = snappy.decompress(Str_UpdatedVertex)
            updated_vertex = np.fromstring(
                Str_UpdatedVertex, dtype=self.__Dtype_All['VertexData'])
            start_id = int(updated_vertex[-1]) * \
                self.__GraphInfo['VertexPerPartition']
            end_id = (int(updated_vertex[-1]) + 1) * \
                self.__GraphInfo['VertexPerPartition']
            if not BSP:
                self.__DataInfo['VertexData'][start_id:end_id] = updated_vertex[
                    0:-1] + self.__DataInfo['VertexData'][start_id:end_id]
                self.__ControlInfo['IterationReport'][int(
                    updated_vertex[-1])] = self.__ControlInfo['IterationReport'][int(updated_vertex[-1])] + 1
            else:
                self.__DataInfo['VertexDataNew'][start_id:end_id] = updated_vertex[
                    0:-1] + self.__DataInfo['VertexData'][start_id:end_id]
                self.__ControlInfo['IterationReport'][int(
                    updated_vertex[-1])] = self.__ControlInfo['IterationReport'][int(updated_vertex[-1])] + 1
                while True:
                    if self.__ControlInfo['IterationNum'] == self.__ControlInfo['IterationReport'].min():
                        break
                    else:
                        sleep(0.1)


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

    def __init__(self, IP, Port, MPIInfo, GraphInfo, Dtype_All):
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
    # __PendingTaskQueue = None
    # __RunningFlag = False
    __stop = threading.Event()

    def stop(self):
        self.__stop.set()

    def __init__(self, DataInfo, GraphInfo, ControlInfo, IP, Port, Dtype_All):
        threading.Thread.__init__(self)
        self.__DataInfo = DataInfo
        self.__GraphInfo = GraphInfo
        self.__ControlInfo = ControlInfo
        self.__Dtype_All = Dtype_All
        self.__IP = IP
        self.__Port = Port
        # self.__PendingTaskQueue = Queue.Queue()
        # __RunningFlag = False
    #
    # def put_task(self, PartitionID):
    #     self.__PendingTaskQueue.put(PartitionID)
    #
    # def is_free(self):
    #     if self.__RunningFlag == False and self.__PendingTaskQueue.empty():
    #         return True

    def run(self):
        while True:
            if self.__stop.is_set():
                break

            if BSP:
                while True:
                    if self.__ControlInfo['IterationNum'] == self.__ControlInfo['IterationReport'].min():
                        break
                    else:
                        sleep(0.1)

            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://%s:%s" % (self.__IP, self.__Port))
            TaskRequest = '1 ' + str(MPI.COMM_WORLD.Get_rank())
            socket.send(TaskRequest)
            message = socket.recv()
            if message == '-1':
                sleep(1)
                continue

            PartitionID_ = int(message)
            UpdatedVertex = self.__ControlInfo['CalcFunc'](
                PartitionID_, self.__DataInfo, self.__GraphInfo, self.__Dtype_All)
            start_id = PartitionID_ * self.__GraphInfo['VertexPerPartition']
            end_id = (PartitionID_ + 1) * \
                self.__GraphInfo['VertexPerPartition']
            UpdatedVertex = UpdatedVertex - \
                self.__DataInfo['VertexData'][start_id:end_id]
            UpdatedVertex[np.where(abs(UpdatedVertex) <= self.__ControlInfo[
                                   'FilterThreshold'])] = 0
            UpdatedVertex = UpdatedVertex.astype(
                self.__Dtype_All['VertexData'])
            Tmp_UpdatedData = np.append(UpdatedVertex, PartitionID_)
            Tmp_UpdatedData = Tmp_UpdatedData.astype(
                self.__Dtype_All['VertexData'])
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

    def __init__(self, IP, Port, MPIInfo, GraphInfo, ControlInfo, Dtype_All):
        threading.Thread.__init__(self)
        self.__IP = IP
        self.__TaskqPort = Port
        self.__MPIInfo = MPIInfo
        self.__GraphInfo = GraphInfo
        self.__ControlInfo = ControlInfo
        self.__Dtype_All = Dtype_All
        self.__stop = threading.Event()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        print(self.__IP, self.__TaskqPort)
        socket.bind("tcp://*:%s" % self.__TaskqPort)

        AllTask = np.zeros(self.__GraphInfo['PartitionNum'], dtype=np.int32)
        # AllTask = AllTask * self.__ControlInfo['MaxIteration']
        AllProgress  = self.__ControlInfo['IterationReport']
        # self.__ControlInfo['StaleNum']
        while True:
            string_receive = socket.recv()
            command, data = string_receive.split()
            if command == '-1':   #exit
                socket.send("-1")
                break
            elif command == '1': #get task
                if AllProgress.min() >= self.__ControlInfo['MaxIteration']:
                    socket.send("-1")
                elif AllTask.min() >= self.__ControlInfo['MaxIteration']:
                    socket.send("-1")
                elif AllProgress.max() - AllProgress.min() <= self.__ControlInfo['StaleNum']:
                    candicate_partition = np.where(AllTask - AllProgress == 0)[0]
                    candicate_status = AllTask[candicate_partition]
                    if len(candicate_partition) == 0:
                        socket.send("-1")
                    else:
                        target_partition = candicate_partition[candicate_status.argmin()]
                        #print target_partition,' to ', data;
                        AllTask[target_partition] = AllTask[target_partition] + 1
                        socket.send(str(target_partition))
                else:
                    socket.send("-1")
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

    def run(self, Str_InitialVertex='zero'):
        self.__MPI_Initial()
        self.__DataInfo['VertexOut'] = load_vertexout(
            self.__GraphInfo, self.__Dtype_All)
        # Initial the vertex data
        self.__DataInfo['VertexData'] = intial_vertex(
            self.__GraphInfo, self.__Dtype_All, Str_InitialVertex)
        if BSP:
            self.__DataInfo['VertexDataNew'] = intial_vertex(
                self.__GraphInfo, self.__Dtype_All, Str_InitialVertex)
        # Communication Thread
        UpdateVertexThread = UpdateThread(
            self.__IP, self.__UpdatePort, self.__MPIInfo, self.__GraphInfo, self.__Dtype_All)
        UpdateVertexThread.start()

        if self.__MPIInfo['MPI_Rank'] == 0:
            TaskSchedulerThread = SchedulerThread(
                self.__IP, self.__TaskqPort, self.__MPIInfo, self.__GraphInfo, self.__ControlInfo, self.__Dtype_All)
            TaskSchedulerThread.start()

        # BroadVertexThread Thread
        BroadVertexThread = BroadThread(
            self.__MPIInfo, self.__DataInfo, self.__ControlInfo, self.__GraphInfo, self.__Dtype_All)
        BroadVertexThread.start()

        TaskThreadPool = []

        for i in range(self.__ThreadNum):
            new_thead = CalcThread(
                self.__DataInfo, self.__GraphInfo, self.__ControlInfo, self.__IP, self.__TaskqPort, self.__Dtype_All)
            TaskThreadPool.append(new_thead)
            new_thead.start()

        if self.__MPIInfo['MPI_Rank'] == 0:
            Old_Vertex_ = self.__DataInfo['VertexData'].copy()

        start_time = time.time()
        end_time   = time.time()

        while True:
            sleep(0.5)
            CurrentIterationNum =  self.__ControlInfo['IterationReport'].min()
            NewIteration = False
            if self.__ControlInfo['IterationNum'] != CurrentIterationNum:
                NewIteration = True
                if BSP:
                    self.__DataInfo['VertexData'] = self.__DataInfo['VertexDataNew'].copy()
                if self.__MPIInfo['MPI_Rank'] == 0:
                    end_time = time.time()
                    print end_time - start_time, ' # Iter: ', CurrentIterationNum, '->', 10000 * LA.norm(self.__DataInfo['VertexData'] - Old_Vertex_)
                    Old_Vertex_ = self.__DataInfo['VertexData'].copy()
                    start_time = time.time()
                self.__ControlInfo['IterationNum'] = CurrentIterationNum

            if CurrentIterationNum == self.__ControlInfo['MaxIteration']:
                break;

        for i in range(self.__ThreadNum):
            TaskThreadPool[i].stop()
        if (self.__MPIInfo['MPI_Rank'] != 0):
            UpdateVertexThread.stop(-1)
        else:
            TaskSchedulerThread.stop(0)
            sleep(0.1)
            UpdateVertexThread.stop(0)
        BroadVertexThread.stop()
        BroadVertexThread.join()
        print "BroadVertexThread->", self.__MPIInfo['MPI_Rank']
        UpdateVertexThread.join()
        print "UpdateVertexThread->", self.__MPIInfo['MPI_Rank']
        if self.__MPIInfo['MPI_Rank'] == 0:
            TaskSchedulerThread.join()


if __name__ == '__main__':
    Dtype_VertexData = np.float32
    Dtype_VertexEdgeInfo = np.int32
    Dtype_EdgeData = np.bool
    Dtype_All = (Dtype_VertexData, Dtype_VertexEdgeInfo, Dtype_EdgeData)

    #DataPath = '/home/mapred/GraphData/wiki/subdata/'
    #VertexNum = 4206800
    #PartitionNum = 20

    DataPath = '/home/mapred/GraphData/uk/subdata/'
    VertexNum = 787803000
    PartitionNum = 3000

    #DataPath = '/home/mapred/GraphData/twitter/subdata/'
    #VertexNum = 41652250
    #PartitionNum = 50

    GraphInfo = (DataPath, VertexNum, PartitionNum, VertexNum/PartitionNum)
    test_graph = satgraph()

    rank_0_host = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        rank_0_host = MPI.Get_processor_name()
    rank_0_host = MPI.COMM_WORLD.bcast(rank_0_host, root=0)

    test_graph.set_Dtype_All(Dtype_All)
    test_graph.set_GraphInfo(GraphInfo)
    test_graph.set_IP(rank_0_host)
    test_graph.set_port(18086, 18087)
    #test_graph.set_ThreadNum(1)
    test_graph.set_ThreadNum(4)
    test_graph.set_MaxIteration(50)
    test_graph.set_StaleNum(3)
    test_graph.set_FilterThreshold(0.000000001)
    test_graph.set_CalcFunc(calc_pagerank)

    test_graph.run('pagerank')
    os._exit(0)
