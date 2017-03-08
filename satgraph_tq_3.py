'''
Created on 14 Apr 2016

@author: sunshine
'''
import logging
import os
import time
import datetime
import numpy as np
import scipy.sparse as sparse
import threading
import Queue
import zmq
import snappy
import ctypes
import gc
import numpy.ctypeslib as npct
from numpy import linalg as LA
from mpi4py import MPI
import sys
import re
import math
import multiprocessing

SLEEP_TIME = 0.2
STOP = False
QueueBroadcastVertex = Queue.Queue()
NP_INF = 10**4
LOCAL_PATH = '/home/mapred/tmp/satgraph'
LIB_PATH = '/home/mapred/share/SatGraph/lib'
array_1d_bool  = npct.ndpointer(dtype=np.bool, ndim=1, flags='CONTIGUOUS')
array_1d_int32 = npct.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')
array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
libsatgraph = npct.load_library("libsatgraph", LIB_PATH)

TOTAL_MEM = 0.0
FREE_MEM = 0.0
MEM_USAGE = 0.0
MEM_CACHE = {}

libsatgraph.add_int32.restype = None
libsatgraph.add_int32.argtypes = [ctypes.c_int32,
                                       array_1d_int32,
                                       array_1d_int32]

libsatgraph.add_float.restype = None
libsatgraph.add_float.argtypes = [ctypes.c_int32,
                                       array_1d_float,
                                       array_1d_float]

libsatgraph.add_double.restype = None
libsatgraph.add_double.argtypes = [ctypes.c_int32,
                                       array_1d_double,
                                       array_1d_double]
 
libsatgraph.minus_int32.restype = None
libsatgraph.minus_int32.argtypes = [ctypes.c_int32,
                                       array_1d_int32,
                                       array_1d_int32]

libsatgraph.minus_float.restype = None
libsatgraph.minus_float.argtypes = [ctypes.c_int32,
                                       array_1d_float,
                                       array_1d_float]

libsatgraph.minus_double.restype = None
libsatgraph.minus_double.argtypes = [ctypes.c_int32,
                                       array_1d_double,
                                       array_1d_double]

libsatgraph.multiply_int32.restype = None
libsatgraph.multiply_int32.argtypes = [ctypes.c_int32,
                                       array_1d_int32,
                                       array_1d_int32]

libsatgraph.multiply_float.restype = None
libsatgraph.multiply_float.argtypes = [ctypes.c_int32,
                                       array_1d_float,
                                       array_1d_float]

libsatgraph.multiply_double.restype = None
libsatgraph.multiply_double.argtypes = [ctypes.c_int32,
                                       array_1d_double,
                                       array_1d_double]
 

libsatgraph.divide_float_int32.restype = None
libsatgraph.divide_float_int32.argtypes = [ctypes.c_int32,
                                           array_1d_float,
                                           array_1d_int32]


libsatgraph.divide_double_int32.restype = None
libsatgraph.divide_double_int32.argtypes = [ctypes.c_int32,
                                            array_1d_double,
                                            array_1d_int32]
 

libsatgraph.ssp_min_int32.restype = None
libsatgraph.ssp_min_int32.argtypes = [array_1d_int32,
                                      array_1d_int32,
                                      ctypes.c_int32,
                                      array_1d_bool,
                                      array_1d_int32,
                                      array_1d_int32]


libsatgraph.ssp_min_float.restype = None
libsatgraph.ssp_min_float.argtypes = [array_1d_int32,
                                      array_1d_int32,
                                      ctypes.c_int32,
                                      array_1d_bool,
                                      array_1d_float,
                                      array_1d_float]


libsatgraph.ssp_min_double.restype = None
libsatgraph.ssp_min_double.argtypes = [array_1d_int32,
                                      array_1d_int32,
                                      ctypes.c_int32,
                                      array_1d_bool,
                                      array_1d_double,
                                      array_1d_double]


libsatgraph.pr_dot_product_float.restype = None
libsatgraph.pr_dot_product_float.argtypes = [array_1d_int32,
                                             array_1d_int32,
                                             ctypes.c_int32,
                                             array_1d_bool,
                                             ctypes.c_int32,
                                             array_1d_int32,
                                             array_1d_float,
                                             array_1d_float,
                                             ctypes.c_int32]

libsatgraph.pr_dot_product_double.restype = None
libsatgraph.pr_dot_product_double.argtypes = [array_1d_int32,
                                             array_1d_int32,
                                             ctypes.c_int32,
                                             array_1d_bool,
                                             ctypes.c_int32,
                                             array_1d_int32,
                                             array_1d_double,
                                             array_1d_double,
                                             ctypes.c_int32]

libsatgraph.component_int32.restype = None
libsatgraph.component_int32.argtypes = [array_1d_int32,
                                       array_1d_int32,
                                       ctypes.c_int32,
                                       array_1d_bool,
                                       array_1d_int32,
                                       array_1d_int32]



libsatgraph.component_float.restype = None
libsatgraph.component_float.argtypes = [array_1d_int32,
                                       array_1d_int32,
                                       ctypes.c_int32,
                                       array_1d_bool,
                                       array_1d_float,
                                       array_1d_float]


libsatgraph.component_double.restype = None
libsatgraph.component_double.argtypes = [array_1d_int32,
                                        array_1d_int32,
                                        ctypes.c_int32,
                                        array_1d_bool,
                                        array_1d_double,
                                        array_1d_double]

def intial_vertex(GraphInfo,
                  Dtype_All,
                  Str_Policy='ones'):
    if Str_Policy == 'ones':
        return np.ones(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
    elif Str_Policy == 'id':
        return np.arange(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
    elif Str_Policy == 'zeros':
        return np.zeros(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
    elif Str_Policy == 'inf':
        temp = np.ones(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
        temp.fill(NP_INF)
        temp = temp.astype(Dtype_All['VertexData'])
        return temp
    elif Str_Policy == 'random':
        temp = np.random.random(GraphInfo['VertexNum'])
        temp = temp.astype(Dtype_All['VertexData'])
        return temp
    elif Str_Policy == 'pagerank':
        temp = np.zeros(GraphInfo['VertexNum'], dtype=Dtype_All['VertexData'])
        temp += 1.0 / GraphInfo['VertexNum']
        temp = temp.astype(Dtype_All['VertexData'])
        return temp
    else:
        return np.ones(GraphInfo['VertexNum'], dtype=Dtype_All[0])


def read_data(path, q):
    f = open(path, 'r')
    data = f.read()
    q.put(data)
    f.close()

def load_edgedata_nodata(PartitionID,
                         GraphInfo,
                         Dtype_All):

    #if MEM_CACHE.has_key(PartitionID):
    #    return MEM_CACHE[PartitionID]

    ###############################################################
    edge_path = GraphInfo['DataPath'] + str(PartitionID) + '.edge'
    ###############################################################
    '''
    hdfs_edge_path = GraphInfo['DataPath'] + str(PartitionID) + '.edge'
    edge_path = LOCAL_PATH + '/' + str(PartitionID) + '.edge'
    if not os.path.exists(edge_path):
        os.system('/opt/hadoop-1.2.1/bin/hadoop ' + \
                  'fs -get ' + \
                  hdfs_edge_path + ' ' + \
                  edge_path)
    
    if not os.path.exists(edge_path):
        print edge_path
        raise Exception('data not found') 
        sys.exit(0)
    '''

    _file = open(edge_path, 'r')

    #q = multiprocessing.Queue()
    #read_process = multiprocessing.Process(target=read_data, args=(edge_path, q,))
    #read_process.start()
    #temp = q.get()
    #read_process.join()
    #temp = np.fromstring(temp, dtype=Dtype_All['VertexEdgeInfo'])
    
    temp = np.fromfile(_file, dtype=Dtype_All['VertexEdgeInfo'])
    indices = temp[5:5 + int(temp[1])]
    indptr = temp[5 + int(temp[1]):5 + int(temp[1]) + int(temp[2])]
    start_id = int(temp[3])
    end_id = int(temp[4])
    _file.close()

    #global FREE_MEM
    #global MEM_USAGE
    #if MEM_USAGE <= 0.7:
    #    MEM_CACHE[PartitionID] = (indices, indptr, end_id-start_id, GraphInfo['VertexNum'], start_id, end_id)
    #    meminfo = open('/proc/meminfo').read()
    #    matched = re.search(r'MemAvailable:\s+(\d+)', meminfo)
    #    FREE_MEM = int(matched.groups()[0])
    #    MEM_USAGE = 1 - 1.0*FREE_MEM/TOTAL_MEM

    return indices, indptr, end_id-start_id, GraphInfo['VertexNum'], start_id, end_id


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
    encoded_shape = (end_id - start_id, GraphInfo['VertexNum'])
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
    # '''
    indices, indptr, shape_0, shape_1, start_id, end_id = \
        load_edgedata_nodata(PartitionID, GraphInfo, Dtype_All)
    ActiveVertex = DataInfo['UpdatedLastIter'][start_id:end_id]
    UpdatedVertex = np.zeros(end_id-start_id+5, dtype=Dtype_All['VertexData'])
    UpdatedVertex[0:-5] = DataInfo['VertexData'][start_id:end_id].copy()
    if len(ActiveVertex) == 0:
        return UpdatedVertex, start_id, end_id

    if Dtype_All['VertexData'] == np.float32:
        libsatgraph.pr_dot_product_float(indices,
                                         indptr,
                                         len(indptr),
                                         ActiveVertex,
                                         end_id-start_id,
                                         DataInfo['VertexOut'],
                                         DataInfo['VertexData'],
                                         UpdatedVertex,
                                         shape_1)

    elif Dtype_All['VertexData'] == np.double:
        libsatgraph.pr_dot_product_double(indices,
                                          indptr,
                                          len(indptr),
                                          ActiveVertex,
                                          end_id-start_id,
                                          DataInfo['VertexOut'],
                                          DataInfo['VertexData'],
                                          UpdatedVertex,
                                          shape_1)
    else:
        raise NameError
    return UpdatedVertex, start_id, end_id

def calc_sssp(PartitionID,
              IterationNum,
              DataInfo,
              GraphInfo,
              Dtype_All):         
    # EdgeMatrix, start_id, end_id = load_edgedata(PartitionID, GraphInfo, Dtype_All)
    indices, indptr, shape_0, shape_1, start_id, end_id = \
        load_edgedata_nodata(PartitionID, GraphInfo, Dtype_All)

    if IterationNum == 0 and PartitionID != 0:
        UpdatedVertex = np.zeros(5, dtype=Dtype_All['VertexData'])
        return UpdatedVertex, 0, 0
    if IterationNum == 0 and PartitionID == 0:
        UpdatedVertex = np.zeros(6, dtype=Dtype_All['VertexData'])
        UpdatedVertex[0] = 0
        return UpdatedVertex, 1, 2

    ActiveVertex = DataInfo['UpdatedLastIter']
    UpdatedVertex = np.zeros(end_id-start_id+5, dtype=Dtype_All['VertexData'])
    UpdatedVertex[0:-5] = DataInfo['VertexData'][start_id:end_id].copy()

    if Dtype_All['VertexData'] == np.float32:
        libsatgraph.ssp_min_float(indices,
                                  indptr,
                                  len(indptr),
                                  ActiveVertex,
                                  DataInfo['VertexData'],
                                  UpdatedVertex)

    elif Dtype_All['VertexData'] == np.double:
        libsatgraph.ssp_min_double(indices,
                                   indptr,
                                   len(indptr),
                                   ActiveVertex,
                                   DataInfo['VertexData'],
                                   UpdatedVertex)
    elif Dtype_All['VertexData'] == np.int32:
        libsatgraph.ssp_min_int32(indices,
                                   indptr,
                                   len(indptr),
                                   ActiveVertex,
                                   DataInfo['VertexData'],
                                   UpdatedVertex)
    else:
        raise NameError

    return UpdatedVertex, start_id, end_id


def calc_component(PartitionID,
              	   IterationNum,
                   DataInfo,
                   GraphInfo,
                   Dtype_All):         
    # EdgeMatrix, start_id, end_id = load_edgedata(PartitionID, GraphInfo, Dtype_All)
    indices, indptr, shape_0, shape_1, start_id, end_id = \
        load_edgedata_nodata(PartitionID, GraphInfo, Dtype_All)
    UpdatedVertex = np.zeros(end_id-start_id+5, dtype=Dtype_All['VertexData'])
    UpdatedVertex[0:-5] = DataInfo['VertexData'][start_id:end_id].copy()
    ActiveVertex = DataInfo['UpdatedLastIter']

    if Dtype_All['VertexData'] == np.float32:
        libsatgraph.component_float(indices,
                                    indptr,
                                    len(indptr),
                                    ActiveVertex,
                                    DataInfo['VertexData'],
                                    UpdatedVertex)

    elif Dtype_All['VertexData'] == np.double:
        libsatgraph.component_double(indices,
                                     indptr,
                                     len(indptr),
                                     ActiveVertex,
                                     DataInfo['VertexData'],
                                     UpdatedVertex)
    elif Dtype_All['VertexData'] == np.int32:
        libsatgraph.component_int32(indices,
                                     indptr,
                                     len(indptr),
                                     ActiveVertex,
                                     DataInfo['VertexData'],
                                     UpdatedVertex)
   
    else:
        raise NameError

    return UpdatedVertex, start_id, end_id

class UpdateBroadcastThread(threading.Thread):
    __MPIInfo = {}
    __DataInfo = None
    __GraphInfo = {}
    __Dtype_All = {}
    __ControlInfo = None
    __stop = None
    __time_a = None
    __time_b = None

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
        self.__time_a = time.time()

    def stop(self):
        self.__stop.set()

    def get_broadcast(self):
        while True:
            try:
                Str_UpdatedVertex = QueueBroadcastVertex.get(block=True, timeout=SLEEP_TIME)
            except:
                continue
            break
        return Str_UpdatedVertex

    def update_BSP(self, updated_vertex, start_id, end_id):
        self.__DataInfo['VertexDataNew'][start_id:end_id] = updated_vertex[0:-5].copy()
        #'''
        if self.__Dtype_All['VertexData'] == np.float32:
            libsatgraph.add_float(end_id-start_id, 
                                  self.__DataInfo['VertexDataNew'][start_id:end_id],
                                  self.__DataInfo['VertexData'][start_id:end_id])
        elif self.__Dtype_All['VertexData'] == np.double:
            libsatgraph.add_double(end_id-start_id, 
                                   self.__DataInfo['VertexDataNew'][start_id:end_id],
                                   self.__DataInfo['VertexData'][start_id:end_id])
        elif self.__Dtype_All['VertexData'] == np.int32:
            libsatgraph.add_int32(end_id-start_id, 
                                   self.__DataInfo['VertexDataNew'][start_id:end_id],
                                   self.__DataInfo['VertexData'][start_id:end_id])
        else:
            raise NameError
        #'''

        #ASYNC
        #self.__DataInfo['VertexData'][start_id:end_id] = self.__DataInfo['VertexDataNew'][start_id: end_id].copy()

        i = int(updated_vertex[-5])
        self.__ControlInfo['IterationReport'][i] += 1

        self.__DataInfo['UpdatedLastIterNew'][start_id:end_id] = updated_vertex[0:-5].copy()
        CurrentIterationNum = self.__ControlInfo['IterationReport'].min()
        if self.__ControlInfo['IterationNum'] != CurrentIterationNum:
            #ASYNC
            self.__DataInfo['VertexData'] = self.__DataInfo['VertexDataNew'].copy()
            self.__DataInfo['UpdatedLastIter'] = self.__DataInfo['UpdatedLastIterNew'].copy()
            self.__DataInfo['UpdatedLastIterNew'].fill(False)
            diff_vertex = self.__DataInfo['UpdatedLastIter'].sum()
            global STOP
            if diff_vertex == 0 and CurrentIterationNum > 5:
                STOP = True
            self.__time_b = time.time()
            if self.__MPIInfo['MPI_Rank'] == 0:
                if (self.__DataInfo['VertexData'] < 0).sum() > 0:
                    print "Error! Check Sync Code first!"
                logging.info('Iter %s, Use %s s, Update %s Vertex', CurrentIterationNum, self.__time_b - self.__time_a, diff_vertex)
            self.__time_a = time.time()

            MPI.COMM_WORLD.Barrier()
            self.__ControlInfo['IterationNum'] = CurrentIterationNum

    def update_broadcast_process(self):
        UpdatedVertex = self.get_broadcast()
        if len(UpdatedVertex) == 4 and UpdatedVertex == 'exit':
            return -1
        UpdatedVertex = snappy.decompress(UpdatedVertex)
        UpdatedVertex = np.fromstring(UpdatedVertex, dtype=self.__Dtype_All['VertexData'])
        start_id = int((UpdatedVertex[-4]) * 10000 + (UpdatedVertex[-3]))
        end_id = int((UpdatedVertex[-2]) * 10000 + (UpdatedVertex[-1]))
        if end_id - start_id != len(UpdatedVertex)-5:
            print "error!"
        self.update_BSP(UpdatedVertex, start_id, end_id)
        return 1

    def run(self):
        while True:
            if self.update_broadcast_process() == -1:
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
            socket_.connect("tcp://%s:%s" % (self.__IP, MPI.COMM_WORLD.Get_rank()+self.__UpdatePort))
            socket_.send("exit")
            socket_.recv()
        else:
            self.__stop.set()

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
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % (MPI.COMM_WORLD.Get_rank()+self.__UpdatePort))
        while True:
            string_receive = socket.recv()
            QueueBroadcastVertex.put(string_receive)
            socket.send("ACK")
            if len(string_receive) == 4 and string_receive == 'exit':
                break

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
        while True:
            if self.__ControlInfo['IterationNum'] == self.__ControlInfo['IterationReport'].min():
                break
            else:
                time.sleep(SLEEP_TIME)
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
                time.sleep(SLEEP_TIME)
                continue
            i = int(message)
            UpdatedVertex, start_id, end_id = \
                self.__ControlInfo['CalcFunc'](i,
                                               self.__ControlInfo['IterationNum'],
                                               self.__DataInfo,
                                               self.__GraphInfo,
                                               self.__Dtype_All)
            UpdatedVertex = UpdatedVertex.astype(self.__Dtype_All['VertexData'])
            np.nan_to_num(UpdatedVertex)
            #'''
            if self.__Dtype_All['VertexData'] == np.float32:
                libsatgraph.minus_float(end_id-start_id, 
                                        UpdatedVertex, 
                                        self.__DataInfo['VertexData'][start_id:end_id])

            elif self.__Dtype_All['VertexData'] == np.double:
                libsatgraph.minus_double(end_id-start_id, 
                                         UpdatedVertex, 
                                         self.__DataInfo['VertexData'][start_id:end_id])

            elif self.__Dtype_All['VertexData'] == np.int32:
                libsatgraph.minus_int32(end_id-start_id, 
                                         UpdatedVertex, 
                                         self.__DataInfo['VertexData'][start_id:end_id])
            else:
                raise NameError
            #'''
            ### @@@@@@@

            UpdatedVertex = UpdatedVertex.astype(self.__Dtype_All['VertexData'])
            UpdatedVertex[-5] = i
            UpdatedVertex[-4] = math.floor(start_id / 10000)
            UpdatedVertex[-3] = start_id % 10000
            UpdatedVertex[-2] = math.floor(end_id / 10000)
            UpdatedVertex[-1] = end_id % 10000
            UpdatedVertex = UpdatedVertex.astype(self.__Dtype_All['VertexData'])
            UpdatedVertex = UpdatedVertex.tostring()
            UpdatedVertex = snappy.compress(UpdatedVertex)

            t_rank = 0
            for host in self.__ControlInfo['Hosts']:
                if MPI.COMM_WORLD.Get_rank() == t_rank:
                    QueueBroadcastVertex.put(UpdatedVertex)
                    t_rank += 1
                    continue
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.connect("tcp://%s:%s" % (host, (1+t_rank+self.__Port)))
                socket.send(UpdatedVertex)
                socket.recv()
                t_rank += 1


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
        if AllProgress.min() >= self.__ControlInfo['MaxIteration'] or STOP:
            socket.send("-1")
        elif AllTask.min() >= self.__ControlInfo['MaxIteration']:
            socket.send("-1")
        elif AllProgress.max() - self.__ControlInfo['IterationNum'] <= 1:
            candicate_partition = np.where(AllTask - AllProgress == 0)[0]
            if len(candicate_partition) == 0:
                socket.send("-1")
            else:
                candicate_status = AllTask[candicate_partition]
                target_status = self.__ControlInfo['IterationNum']
                target_ids = np.where(candicate_status == target_status)[0]
                if len(target_ids) == 0:
                    socket.send("-1")
                else:
                    target_partition = candicate_partition[target_ids]
                    target_locality = \
                        LocalityInfo[self.__ControlInfo['Hosts'][rank]][target_partition]
                    max_allocate = target_locality.argmax()
                    target_partition = target_partition[max_allocate]
                    # self.__ControlInfo['IterationNum']
                    AllTask[target_partition] += 1
                    LocalityInfo[self.__ControlInfo['Hosts'][rank]][target_partition] += 1
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
        # for i in xrange(MPI.COMM_WORLD.Get_size()):
        all_hosts = self.__ControlInfo['Hosts']
        all_hosts = list(set(all_hosts))
        for i in all_hosts:
            LocalityInfo[i] = np.zeros(self.__GraphInfo['PartitionNum'],
                                       dtype=np.int32)

        while True:
            string_receive = socket.recv()
            command, rank = string_receive.split()
            if command == '-1':  # exit
                socket.send("-1")
                break
            elif command == '1':  # get task
                rank = int(rank)
                first_task_time = self.assign_task(rank, LocalityInfo, AllTask, AllProgress, socket)
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
        self.__ControlInfo['FilterThreshold'] = 0
        self.__ControlInfo['CalcFunc'] = None
        self.__ControlInfo['Hosts'] = None
        self.__DataInfo['EdgeData'] = {}
        self.__DataInfo['VertexOut'] = None
        self.__DataInfo['VertexIn'] = None
        self.__DataInfo['VertexData'] = None
        self.__DataInfo['VertexDataNew'] = None
        self.__DataInfo['UpdatedLastIter'] = None
        self.__DataInfo['UpdatedLastIterNew'] = None

    def set_FilterThreshold(self, FilterThreshold):
        self.__ControlInfo['FilterThreshold'] = FilterThreshold

    def set_CalcFunc(self, CalcFunc):
        self.__ControlInfo['CalcFunc'] = CalcFunc

    def set_ThreadNum(self, ThreadNum):
        self.__ThreadNum = ThreadNum

    def set_Hosts(self, Hosts):
        self.__ControlInfo['Hosts'] = Hosts

    def set_MaxIteration(self, MaxIteration):
        self.__ControlInfo['MaxIteration'] = MaxIteration

    def set_port(self, Port1):
        self.__UpdatePort = Port1+1
        self.__TaskqPort = Port1

    def set_IP(self, IP):
        self.__IP = IP

    def set_GraphInfo(self, GraphInfo):
        self.__GraphInfo['DataPath'] = GraphInfo[0]
        self.__GraphInfo['VertexNum'] = GraphInfo[1]
        self.__GraphInfo['PartitionNum'] = GraphInfo[2]
        self.__ControlInfo['IterationReport'] = np.zeros(self.__GraphInfo['PartitionNum'], dtype=np.uint16)

    def set_Dtype_All(self, Dtype_All):
        self.__Dtype_All['VertexData'] = Dtype_All[0]
        self.__Dtype_All['VertexEdgeInfo'] = Dtype_All[1]
        self.__Dtype_All['EdgeData'] = Dtype_All[2]

    def __MPI_Initial(self):
        self.__MPIInfo['MPI_Comm'] = MPI.COMM_WORLD
        self.__MPIInfo['MPI_Size'] = self.__MPIInfo['MPI_Comm'].Get_size()
        self.__MPIInfo['MPI_Rank'] = self.__MPIInfo['MPI_Comm'].Get_rank()

    def graph_process(self, Iteration):
        CurrentIterationNum = self.__ControlInfo['IterationNum']
        NewIteration = False
        if self.__ControlInfo['IterationNum'] != Iteration:
            NewIteration = True
        return NewIteration, CurrentIterationNum

    def create_threads(self):
        UpdateVertexThread = \
            UpdateThread(self.__IP,
                         self.__UpdatePort,
                         self.__MPIInfo,
                         self.__GraphInfo,
                         self.__Dtype_All)
        UpdateVertexThread.start()

        TaskSchedulerThread = None
        if self.__MPIInfo['MPI_Rank'] == 0:
            TaskSchedulerThread = SchedulerThread(self.__IP,
                                                  self.__TaskqPort,
                                                  self.__MPIInfo,
                                                  self.__GraphInfo,
                                                  self.__ControlInfo,
                                                  self.__Dtype_All)
            TaskSchedulerThread.start()

        UpdateBstThread = UpdateBroadcastThread(self.__MPIInfo,
                                                      self.__DataInfo,
                                                      self.__ControlInfo,
                                                      self.__GraphInfo,
                                                      self.__Dtype_All)
        UpdateBstThread.start()

        MPI.COMM_WORLD.Barrier()

        TaskThreadPool = []

        
        #if self.__MPIInfo['MPI_Rank'] != 0:
        for i in range(self.__ThreadNum):
            new_task_thead = CalcThread(self.__DataInfo,
                                            self.__GraphInfo,
                                            self.__ControlInfo,
                                            self.__IP,
                                            self.__TaskqPort,
                                            self.__Dtype_All)
            TaskThreadPool.append(new_task_thead)
            new_task_thead.start()
        return UpdateVertexThread, TaskSchedulerThread, UpdateBstThread, TaskThreadPool

    def destroy_threads(self,
                        UpdateVertexThread,
                        TaskSchedulerThread,
                        UpdateBstThread,
                        TaskThreadPool):

        #if self.__MPIInfo['MPI_Rank'] != 0:
        for i in range(self.__ThreadNum):
            TaskThreadPool[i].stop()

        if (self.__MPIInfo['MPI_Rank'] != 0):
            UpdateVertexThread.stop(-1)
        else:
            TaskSchedulerThread.stop(0)
            time.sleep(1)
            UpdateVertexThread.stop(0)
        UpdateBstThread.stop()

        UpdateVertexThread.join()
        UpdateBstThread.join()
        if self.__MPIInfo['MPI_Rank'] == 0:
            TaskSchedulerThread.join()

    def run(self, InitialVertex='zero'):
        self.__MPI_Initial()
        self.__DataInfo['VertexOut'] = load_vertexout(self.__GraphInfo,
                                                      self.__Dtype_All)
        self.__DataInfo['VertexData'] = intial_vertex(self.__GraphInfo,
                                                      self.__Dtype_All,
                                                      InitialVertex)
        self.__DataInfo['VertexDataNew'] = self.__DataInfo['VertexData'].copy()
        self.__DataInfo['UpdatedLastIter'] = np.ones(self.__GraphInfo['VertexNum'], dtype=np.bool)
        self.__DataInfo['UpdatedLastIterNew'] = np.zeros(self.__GraphInfo['VertexNum'], dtype=np.bool)
        UpdateVertexThread, TaskSchedulerThread, UpdateBstThread, TaskThreadPool = self.create_threads()
        start_time = time.time()
        app_start_time = time.time()
        Iteration = 0
        global STOP

        while True:
            time.sleep(10)
            if self.__ControlInfo['IterationNum'] >= self.__ControlInfo['MaxIteration'] or STOP:
                break
        if self.__MPIInfo['MPI_Rank'] == 0:
            app_end_time = time.time()
            print 'Time Used: ', app_end_time - app_start_time

        MPI.COMM_WORLD.Barrier()
        self.destroy_threads(UpdateVertexThread,
                             TaskSchedulerThread,
                             UpdateBstThread,
                             TaskThreadPool)

if __name__ == '__main__':

    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(8)))

    logging.getLogger().setLevel(logging.INFO)
    #Dtype_VertexData = np.double
    #Dtype_VertexData = np.float32
    Dtype_VertexData = np.int32

    Dtype_VertexEdgeInfo = np.int32
    Dtype_EdgeData = np.bool
    Dtype_All = (Dtype_VertexData, Dtype_VertexEdgeInfo, Dtype_EdgeData)


    meminfo = open('/proc/meminfo').read()
    matched = re.search(r'^MemTotal:\s+(\d+)', meminfo)
    TOTAL_MEM = int(matched.groups()[0])
    meminfo = open('/proc/meminfo').read()
    matched = re.search(r'MemAvailable:\s+(\d+)', meminfo)
    FREE_MEM = int(matched.groups()[0])
    MEM_USAGE = 1 - 1.0*FREE_MEM/TOTAL_MEM

    #
    #DataPath = '/home/mapred/GraphData/uk/edge3/'
    #VertexNum = 787803000
    #PartitionNum = 2379

    #DataPath = '/home/mapred/GraphData/soc/edge2/'
    #VertexNum = 4847571
    #PartitionNum = 14

    #DataPath = '/home/mapred/GraphData/twitter/edge2/'
    #VertexNum = 41652250
    #PartitionNum = 294

    DataPath = '/home/mapred/GraphData/webuk_3/'
    VertexNum = 133633040
    PartitionNum = 300

    #DataPath = '/home/mapred/GraphData/eu/edge/'
    #VertexNum = 1070560000
    #PartitionNum = 5096
    
    GraphInfo = (DataPath, VertexNum, PartitionNum)
    test_graph = satgraph()

    rank_0_host = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        rank_0_host = MPI.Get_processor_name()
    rank_0_host = MPI.COMM_WORLD.bcast(rank_0_host, root=0)
    all_hosts = MPI.Get_processor_name()
    all_hosts = MPI.COMM_WORLD.gather(all_hosts, root=0)
    all_hosts = MPI.COMM_WORLD.bcast(all_hosts, root=0)
    test_graph.set_Hosts(all_hosts)

    #'''
    host_to_rank = {}
    j = 0
    for i in all_hosts:
        host_to_rank[i] = j
        j += 1

    if (MPI.COMM_WORLD.Get_rank() in host_to_rank.values()):
        d = os.path.dirname(LOCAL_PATH)
        if not os.path.exists(d):
            os.mkdir(LOCAL_PATH)
        else:
            os.system('rm -rf ' + LOCAL_PATH)
            os.mkdir(LOCAL_PATH)
    MPI.COMM_WORLD.Barrier()
    #'''

    test_graph.set_Dtype_All(Dtype_All)
    test_graph.set_GraphInfo(GraphInfo)
    test_graph.set_IP(rank_0_host)
    test_graph.set_port(18086)
    test_graph.set_ThreadNum(12)
    test_graph.set_MaxIteration(20)
    #test_graph.set_FilterThreshold(0.0001/VertexNum)
    test_graph.set_FilterThreshold(0)
    
    test_graph.set_CalcFunc(calc_sssp)
    #test_graph.set_CalcFunc(calc_pagerank)
    #test_graph.set_CalcFunc(calc_component)

    MPI.COMM_WORLD.Barrier()
    
    test_graph.run('inf')
    #test_graph.run('pagerank')
    #test_graph.run('id')
    os._exit(0)
