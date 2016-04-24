'''
Created on 14 Apr 2016

@author: sunshine
'''
import os
import numpy as np
import scipy.sparse as sparse
from mpi4py import MPI
import shutil
import math
import threading
import Queue
import zmq
from time import sleep

QueueUpdatedVertex = Queue.Queue();

def preprocess_graph(Str_RawDataPath, 
                     Str_DestDataPath, 
                     Str_Seq='\t'):
    
    if not os.path.isfile(Str_RawDataPath):
        return -1;
    _File_RawData = open(Str_RawDataPath, 'r');
    _Dict_Map = {};#can also be levelDB
    _Processed_Data = [];
    _Int_InitialNum = -1;
    while True:
        _Str_Line = _File_RawData.readline()
        if len(_Str_Line) == 0:
            break;
        if _Str_Line[0] == '#':
            continue;
        _Str_Temp = _Str_Line.split(Str_Seq);
        if len(_Str_Temp) != 2:
            continue;
        try:
            _Int_i = int(_Str_Temp[0]);
            _Int_j = int(_Str_Temp[1]);
            _Int_Mapped_i = 0;
            _Int_Mapped_j = 0;
            if _Int_i in _Dict_Map:
                _Int_Mapped_i = _Dict_Map[_Int_i];
            else:
                _Int_InitialNum = _Int_InitialNum + 1;
                _Dict_Map[_Int_i] = _Int_InitialNum;
                _Int_Mapped_i = _Int_InitialNum;
                
            if _Int_j in _Dict_Map:
                _Int_Mapped_j = _Dict_Map[_Int_j];
            else:
                _Int_InitialNum = _Int_InitialNum + 1;
                _Dict_Map[_Int_j] = _Int_InitialNum;
                _Int_Mapped_j = _Int_InitialNum;
            _Processed_Data.append((_Int_Mapped_i, _Int_Mapped_j));
        except:
            print 'Cannot format Data ', _Str_Line;

    _File_RawData.close();
    
    _File_DestData = open(Str_DestDataPath, 'w');
    for i in _Processed_Data:
        _File_DestData.write(str(i[0])+'\t' + str(i[1]) +'\n');
    _File_DestData.close();
    
    return len(_Dict_Map), len(_Processed_Data);

def graph_to_matrix(Str_RawDataPath, 
                    Str_DestDataPath, 
                    Int_VertexNum, 
                    Int_PartitionNum, 
                    Dtype_All, 
                    Str_Seq = '\t'):
    
    if not os.path.isfile(Str_RawDataPath):
        return -1;
    
    Int_VertexPerPartition = int(math.ceil(Int_VertexNum*1.0/Int_PartitionNum));
    Int_NewVertexNum = Int_PartitionNum * Int_VertexPerPartition;
    
    '''
    Define the matrix, use lil matrix first
    '''
    _SMat_EdgeData = sparse.lil_matrix((Int_NewVertexNum, Int_NewVertexNum), dtype=Dtype_All[2]);
    _Array_VertexIn = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1]);
    _Array_VertexOut = np.zeros(Int_NewVertexNum, dtype=Dtype_All[1]);
    _File_RawData = open(Str_RawDataPath, 'r');
    _Str_Line = '';
    
    '''
    Read data from the raw data file
    '''
    while True:
        _Str_Line = _File_RawData.readline()
        if len(_Str_Line) == 0:
            break;
        if _Str_Line[0] == '#':
            continue;
        _Str_Temp = _Str_Line.split(Str_Seq);
        if len(_Str_Temp) != 2:
            continue;
        try:
            _Int_i = int(_Str_Temp[0]);
            _Int_j = int(_Str_Temp[1]);
        except:
            print 'Cannot format Data ', _Str_Line;
        '''
        j is dst, i is src
        '''
        _SMat_EdgeData[_Int_j, _Int_i] = 1;
        _Array_VertexIn[_Int_j]  = _Array_VertexIn[_Int_j] + 1;
        _Array_VertexOut[_Int_i] = _Array_VertexOut[_Int_i] + 1;
    _File_RawData.close();
    
    _Array_VertexOut[np.where(_Array_VertexOut == 0)] = 1;
    _Array_VertexIn[np.where(_Array_VertexIn == 0)] = 1;
    _SMat_EdgeData = _SMat_EdgeData.tocsr();
    
    if os.path.isfile(Str_DestDataPath+'/subdata'):
        os.remove(Str_DestDataPath+'/subdata');
    if os.path.isdir(Str_DestDataPath+'/subdata'):
        shutil.rmtree(Str_DestDataPath+'/subdata');
    os.makedirs(Str_DestDataPath+'/subdata');
    
    for i in range(Int_PartitionNum):
        _File_PartitionData = open(Str_DestDataPath+'/subdata/'+str(i)+'.edge', 'w');
        Partition_SMat_EdgeData = _SMat_EdgeData[i*Int_VertexPerPartition: (i+1)*Int_VertexPerPartition];
        Partition_Indices = Partition_SMat_EdgeData.indices;
        Partition_Indptr = Partition_SMat_EdgeData.indptr;
        Len_Indices = len(Partition_Indices);
        Len_Indptr= len(Partition_Indptr);
        PartitionData = np.append(len(Partition_SMat_EdgeData.data), Len_Indices);
        PartitionData = np.append(PartitionData, Len_Indptr);
        PartitionData = np.append(PartitionData, Partition_Indices);
        PartitionData = np.append(PartitionData, Partition_Indptr);
        PartitionData = PartitionData.astype(Dtype_All[1]);
        PartitionData.tofile(_File_PartitionData);
        _File_PartitionData.close();
        
    _File_PartitionData = open(Str_DestDataPath+'/subdata/'+'vertexout', 'w');
    _Array_VertexOut.tofile(_File_PartitionData);
    _File_PartitionData.close();
    _File_PartitionData = open(Str_DestDataPath+'/subdata/'+'vertexin', 'w');
    _Array_VertexIn.tofile(_File_PartitionData);
    _File_PartitionData.close();
        
    return Str_DestDataPath+'/subdata/', Int_NewVertexNum, Int_PartitionNum, Int_VertexPerPartition

def intial_vertex(GraphInfo, 
                  Dtype_All,
                  Str_Policy = 'ones'):
    if Str_Policy == 'ones':
        return np.ones(GraphInfo[1], dtype = Dtype_All[0]);
    elif Str_Policy == 'zeros':
        return np.zeros(GraphInfo[1], dtype = Dtype_All[0]);
    elif Str_Policy == 'random':
        temp = np.random.random(GraphInfo[1]);
        temp = temp.astype(Dtype_All[0]);
        return temp;
    elif Str_Policy == 'pagerank':
        temp = np.zeros(GraphInfo[1], dtype=Dtype_All[0]);
        temp = temp + 1.0/GraphInfo[1];
        temp = temp.astype(Dtype_All[0])
        return temp;
    else:
        return np.ones(GraphInfo[1], dtype = Dtype_All[0]);
             
def load_edgedata(PartitionID, GraphInfo, Dtype_All):
    _file = open(GraphInfo[0] + str(PartitionID) + '.edge', 'r');
    temp = np.fromfile(_file, dtype=Dtype_All[1]);
    data = np.ones(temp[0], dtype=Dtype_All[2]);
    indices = temp[3:3+int(temp[1])];
    indptr  = temp[3+int(temp[1]):3+int(temp[1])+int(temp[2])];
    #csr_matrix((data, indices, indptr), [shape=(M, N)])
    mat_data = sparse.csr_matrix((data, indices, indptr), shape=(GraphInfo[3], GraphInfo[1]));
    _file.close();
    return mat_data;

def load_vertexin(GraphInfo, Dtype_All):
    _file = open(GraphInfo[0] + 'vertexin', 'r');
    temp = np.fromfile(_file, dtype = Dtype_All[1]);
    _file.close();
    return temp;

def load_vertexout(GraphInfo, Dtype_All):
    _file = open(GraphInfo[0] + 'vertexout', 'r');
    temp = np.fromfile(_file, dtype = Dtype_All[1]);
    _file.close();
    return temp;


#     DataPath = './subdata/'
#     VertexNum = 81310;
#     PartitionNum = 10;
#     VertexPerPartition = 8131;
#     GraphInfo = (DataPath, VertexNum, PartitionNum, VertexPerPartition);

def calc_pagerank(EdgeData, VertexOut, VertexIn, VertexData, GraphInfo, Dtype_All):
    UpdatedVertex = EdgeData.dot(VertexData/VertexOut) * 0.85 + 1.0/GraphInfo[1];
    UpdatedVertex = UpdatedVertex.astype(Dtype_All[0]);
    return UpdatedVertex;

class BroadThread(threading.Thread):
    __MPI_Comm = None;
    __MPI_Size = 0;
    __MPI_Rank = 0;
    __VertexData = None;
    __GraphInfo = [];
#     __UpdatedVertex = None;
    __Dtype_All = [];
    __IterationReport = None;
    
    
    def __init__(self, MPI_Comm, MPI_Size, MPI_Rank, VertexData, ProgressReport, GraphInfo, Dtype_All):
        threading.Thread.__init__(self);
        self.__MPI_Comm = MPI_Comm;
        self.__MPI_Size = MPI_Size;
        self.__MPI_Rank = MPI_Rank;
        self.__VertexData = VertexData;
        self.__IterationReport = ProgressReport;
        self.__GraphInfo  = GraphInfo;
        self.__Dtype_All  = Dtype_All;
    
    def run(self):
        while True:
            if self.__MPI_Rank == 0:           
                Str_UpdatedVertex = None;
                Str_UpdatedVertex = QueueUpdatedVertex.get();
            else:
                Str_UpdatedVertex = None;
            
            self.__MPI_Comm.barrier();    
            Str_UpdatedVertex = self.__MPI_Comm.bcast(Str_UpdatedVertex, root=0);
            updated_vertex = np.fromstring(Str_UpdatedVertex, dtype = self.__Dtype_All[0]);
            start_id = int(updated_vertex[-1]) * self.__GraphInfo[3];
            end_id   = (int(updated_vertex[-1])+1) * self.__GraphInfo[3];
            self.__VertexData[start_id:end_id] = updated_vertex[0:-1];
            self.__IterationReport[int(updated_vertex[-1])] = self.__IterationReport[int(updated_vertex[-1])] + 1;
#             if(self.__MPI_Rank == 0):
#                 print start_id, end_id, updated_vertex.dtype;
            

class UpdateThread(threading.Thread):
    __MPI_Size = 0;
    __MPI_Rank = 0;
    __VertexData = None;
    __GraphInfo = [];
    __IP = '127.0.0.1';
    __Port = 17070;
    __Dtype_All = [];
    
    def __init__(self, IP, Port, MPI_Size, MPI_Rank, VertexData, GraphInfo, Dtype_All):
        threading.Thread.__init__(self);
        self.__IP = IP;
        self.__Port = Port;
        self.__MPI_Size = MPI_Size;
        self.__MPI_Rank = MPI_Rank;
        self.__VertexData = VertexData;
        self.__GraphInfo  = GraphInfo;
        self.__Dtype_All  = Dtype_All;
    
    def run(self):
        if self.__MPI_Rank == 0:
            context = zmq.Context();
            socket = context.socket(zmq.REP);
            print (self.__IP, self.__Port);
            socket.bind("tcp://*:%s" % self.__Port);
            while True:
                string_receive = socket.recv();
                QueueUpdatedVertex.put(string_receive);
                socket.send("OK");
                
        else:
            while True:
                Str_UpdatedVertex =  QueueUpdatedVertex.get();
                context = zmq.Context();
                socket = context.socket(zmq.REQ);
                socket.connect("tcp://%s:%s" % (self.__IP, self.__Port));
                socket.send(Str_UpdatedVertex);
                socket.recv();

class CalcThread(threading.Thread):
    __partitionID = 0;
    __EdgeData = [];
    __VertexOut = [];
    __VertexIn = [];
    __VertexData = [];
    __GraphInfo = [];
    __CalcFunc = None;
    __Dtype_All = [];
    
    def __init__(self, partitionID, EdgeData, VertexOut, VertexIn, VertexData, GraphInfo, CalcFunc, Dtype_All):
        threading.Thread.__init__(self);
        self.__partitionID = partitionID;
        self.__EdgeData    = EdgeData;
        self.__VertexOut   = VertexOut;
        self.__VertexIn    = VertexIn;
        self.__VertexData  = VertexData;
        self.__GraphInfo   = GraphInfo;
        self.__CalcFunc    = CalcFunc;
        self.__Dtype_All   = Dtype_All;
        
    def run(self):
        '''
        pagerank as example
        '''
        UpdatedVertex = self.__CalcFunc(self.__EdgeData, 
                                        self.__VertexOut, 
                                        self.__VertexIn, 
                                        self.__VertexData, 
                                        self.__GraphInfo,
                                        self.__Dtype_All);
        Tmp_UpdatedData = np.append(UpdatedVertex, self.__partitionID);
        Tmp_UpdatedData = Tmp_UpdatedData.astype(self.__Dtype_All[0]);
        Str_UpdatedData = Tmp_UpdatedData.tostring();
        QueueUpdatedVertex.put(Str_UpdatedData);
#         print QueueUpdatedVertex.qsize();

class satgraph():
    __Dtype_VertexData      = np.int32;
    __Dtype_VertexEdgeInfo  = np.int32;
    __Dtype_EdgeData        = np.int32;
    __Dtype_All             = (np.int32, np.int32, np.int32);
    __DataPath = './subdata/';
    __VertexNum = 0;
    __PartitionNum = 0;
    __VertexPerPartition = 0;
    __GraphInfo = ('./subdata/', 0, 0, 0);
    __MPI_Comm = None;
    __MPI_Comm_Update = None;
    __MPI_Size = None;
    __MPI_Rank = None;
    __PartitionInfo = {};
    __EdgeData  = {};
    __VertexOut = [];
    __VertexIn = [];
    __VertexData = [];
    __Port = 17070;
    __IP   = '127.0.0.1';
    __IterationNum = 0;
    __IterationReport = None;
    __ThreadNum = 1;
    __MaxIteration = 10;
    __CalcFunc = None;
    __StaleNum = 0;
    
    def __init__(self):
        self.__Dtype_VertexData      = np.int32;
        self.__Dtype_VertexEdgeInfo  = np.int32;
        self.__Dtype_EdgeData        = np.int32;
        self.__Dtype_All             = (np.int32, np.int32, np.int32);
        self.__DataPath = './subdata/';
        self.__VertexNum = 0;
        self.__PartitionNum = 0;
        self.__VertexPerPartition = 0;
        self.__GraphInfo = ('./subdata/', 0, 0, 0);
        self.__Port = 17070;
    
    def set_StaleNum(self, StaleNum):
        self.__StaleNum = StaleNum; 
    
    def set_CalcFunc(self, CalcFunc):
        self.__CalcFunc = CalcFunc;
        
    def set_ThreadNum(self, ThreadNum):
        self.__ThreadNum = ThreadNum;        
        
    def set_MaxIteration(self, MaxIteration):
        self.__MaxIteration = MaxIteration;
    
    def set_port(self, Port):
        self.__Port = Port;
        
    def set_IP(self, IP):
        self.__IP = IP;
        
    def set_GraphInfo(self, GraphInfo):
        self.__GraphInfo = GraphInfo;
        self.__DataPath  = GraphInfo[0];
        self.__VertexNum = GraphInfo[1];
        self.__PartitionNum = GraphInfo[2];
        self.__VertexPerPartition = GraphInfo[3];
        self.__IterationReport = np.zeros(GraphInfo[2]);
    
    def set_Dtype_All(self, Dtype_All):
        self.__Dtype_All = Dtype_All;
        self.__Dtype_VertexData      = Dtype_All[0];
        self.__Dtype_VertexEdgeInfo  = Dtype_All[1];
        self.__Dtype_EdgeData        = Dtype_All[2];
    
    def set_Dtype_VertexData(self, Dtype_VertexData):
        self.__Dtype_VertexData = Dtype_VertexData;
        self.__Dtype_All = (self.__Dtype_VertexData, 
                            self.__Dtype_VertexEdgeInfo, 
                            self.__Dtype_EdgeData);
        
    def set_Dtype_VertexEdgeInfo(self, Dtype_VertexEdgeInfo):
        self.__Dtype_VertexEdgeInfo = Dtype_VertexEdgeInfo;
        self.__Dtype_All = (self.__Dtype_VertexData, 
                            self.__Dtype_VertexEdgeInfo, 
                            self.__Dtype_EdgeData);
        
    def set_Dtype_EdgeData (self, Dtype_EdgeData ):
        self.__Dtype_EdgeData  = Dtype_EdgeData;
        self.__Dtype_All = (self.__Dtype_VertexData, 
                            self.__Dtype_VertexEdgeInfo, 
                            self.__Dtype_EdgeData);
     
    @property
    def CalcFunc(self):
        return self.__CalcFunc;
    
    @property
    def StaleNum(self):
        return self.__StaleNum;
       
    @property
    def Port(self):
        return self.__Port;
    
    @property
    def MaxIteration(self):
        return self.__MaxIteration;
        
    @property
    def ThreadNum(self):
        return self.__ThreadNum;
    
    @property
    def IP(self):
        return self.__IP;
    
    @property
    def Dtype_All(self):
        return self.__Dtype_All;
    
    @property
    def GraphInfo(self):
        return self.__GraphInfo;

    @property
    def MPI_Size(self):
        return self.__MPI_Size;
    
    @property
    def MPI_Rank(self):
        return self.__MPI_Rank;
    
    @property
    def PartitionInfo(self):
        return self.__PartitionInfo;
    
    @property
    def EdgeData(self):
        return self.__EdgeData;
    
    @property
    def VertexOut(self):
        return self.__VertexOut;
    
    @property
    def VertexIn(self):
        return self.__VertexIn;
    
    @property
    def VertexData(self):
        return self.__VertexData;
    
    def run(self, Str_VertexType):

        self.__MPI_Comm = MPI.COMM_WORLD;
        self.__MPI_Size = self.__MPI_Comm.Get_size();
        self.__MPI_Rank = self.__MPI_Comm.Get_rank();
        
        self.__MPI_Comm_Update = MPI.Comm.Dup(self.__MPI_Comm);
        '''
        Initial the PartitionInfo
        '''
        PartitionPerNode = int(math.floor(PartitionNum*1.0/self.__MPI_Size));
        
        for i in range(self.__MPI_Size):
            if i == self.__MPI_Size-1:
                self.__PartitionInfo[i] = range(i*PartitionPerNode, PartitionNum);
            else:
                self.__PartitionInfo[i] = range(i*PartitionPerNode, (i+1)*PartitionPerNode);   
        
        '''
        load data to cache
        '''
        for i in self.__PartitionInfo[self.__MPI_Rank]:
            self.__EdgeData[i]  = load_edgedata(i, self.__GraphInfo, self.__Dtype_All);
        self.__VertexOut = load_vertexout(self.__GraphInfo, self.__Dtype_All);
        '''
        Initial the vertex data
        '''
        self.__VertexData = intial_vertex(self.__GraphInfo, self.__Dtype_All, Str_VertexType);
        
        #ID, MPI_Comm, MPI_Size, MPI_Rank, VertexData, GraphInfo, Dtype_All
        '''
        Communication Threads
        '''
        UpdateVertexThread = UpdateThread(self.__IP, self.__Port, 
                                          self.__MPI_Size, self.__MPI_Rank, 
                                          self.__VertexData, self.__GraphInfo,
                                          self.__Dtype_All);
        UpdateVertexThread.start();
        
        BroadVertexThread = BroadThread(self.__MPI_Comm, self.__MPI_Size, 
                     self.__MPI_Rank, self.__VertexData, self.__IterationReport,
                     self.__GraphInfo,self.__Dtype_All);
        BroadVertexThread.start();
        
        
        AllTaskQueue = Queue.Queue();
        for i in range(self.__MaxIteration):
            for j in self.__PartitionInfo[self.__MPI_Rank]:
                AllTaskQueue.put(j);
        
        def check_threadpool(threadpool):
            for i in threadpool:
                if i.is_alive():
                    pass;
                else:
                    threadpool.remove(i);
            return len(threadpool);
        TaskThreadPool = [];
        TaskTotalNum = 0;
        
        while not AllTaskQueue.empty():           
            while True:
                running_num =  check_threadpool(TaskThreadPool);
                if  running_num < self.__MaxIteration:
                    break;
                else:
                    sleep(0.01);
            new_partion =  AllTaskQueue.get();
            new_thead = CalcThread(new_partion, 
                             self.__EdgeData[new_partion], 
                             self.__VertexOut, 
                             self.__VertexIn, 
                             self.__VertexData, 
                             self.__GraphInfo,
                             self.__CalcFunc,
                             self.__Dtype_All); 
            
            CurrentIterationNum = TaskTotalNum/len(self.__PartitionInfo[self.__MPI_Rank]);
            NewIteration = False;
            if self.__IterationNum != CurrentIterationNum:
                self.__IterationNum = CurrentIterationNum;
                NewIteration = True;
            TaskTotalNum = TaskTotalNum + 1;
            
            while True:
                if (self.__IterationNum - self.__IterationReport.min()) <= self.__StaleNum:
                    break;
                sleep(0.01);
            new_thead.start();
            TaskThreadPool.append(new_thead);
            
            if NewIteration:
                if self.__MPI_Rank == 0:
                    print self.__VertexData;

if __name__ == '__main__':
    Dtype_VertexData      = np.float32;
    Dtype_VertexEdgeInfo  = np.int32;
    Dtype_EdgeData        = np.int8;
    Dtype_All = (Dtype_VertexData, Dtype_VertexEdgeInfo, Dtype_EdgeData);    
    
    DataPath = './subdata/'
    VertexNum = 81310;
    PartitionNum = 10;
    VertexPerPartition = 8131;
    GraphInfo = (DataPath, VertexNum, PartitionNum, VertexPerPartition);
    
    test_graph = satgraph();
    test_graph.set_Dtype_All(Dtype_All);
    test_graph.set_GraphInfo(GraphInfo);
    test_graph.set_IP('localhost');
    test_graph.set_port(18085);
    test_graph.set_ThreadNum(1);
    test_graph.set_MaxIteration(20);
    test_graph.set_StaleNum(2);
    test_graph.set_CalcFunc(calc_pagerank);
    
#    a = preprocess_graph('./twitter.txt', './twitter2.txt', ' ');
#     GraphInfo = \
#         graph_to_matrix('./twitter2.txt', './', 81306, 10, Dtype_All);
    #81310, 8131, 10    
    test_graph.run('pagerank') 
#     print PartitionInfo[MPI_Rank];
    pass