from __future__ import print_function
import numpy as np
import math
import scipy.sparse as sparse
from pyspark import SparkConf, SparkContext
from pyspark.storagelevel import StorageLevel
from multiprocessing.dummy import Pool as ThreadPool 
import gc
import re
import subprocess
import sys
print = lambda x: sys.stdout.write("%s\n" % x)
tmp_path = '/home/mapred/tmp/'

partition_num = 300
vertex_num = 41652250
data_path = "hdfs://bdp-10:9000/tsv/twitter-2010.tsv"

#partition_num = 2500
#vertex_num = 787803000
#data_path = "hdfs://bdp-10:9000/tsv/uk-2014.tsv"

#partition_num = 20
#vertex_num = 4847571
#data_path = "hdfs://bdp-10:9000/tsv/soc.tsv"

subprocess.Popen(["/opt/hadoop-1.2.1/bin/hadoop", "fs", "-rmr", "/out_convert"], stdout=subprocess.PIPE)
subprocess.Popen(["rm", tmp_path+"part-*"], stdout=subprocess.PIPE)
splitter = '\t'
round_num = 10

conf = (SparkConf().setAppName("SatGraph Convert"))
sc = SparkContext(conf = conf)

def f1(line):
    a,b = line.split()
    return int(b), 1

lines = sc.textFile(data_path)
# counts = lines.map(f1).reduceByKey(lambda a, b: a+b)
counts = lines.map(f1).reduceByKey(lambda a, b: a+b, numPartitions=100)
counts.saveAsTextFile("hdfs://bdp-10:9000/out_convert")

print ("############# Constructing Array ##########")
pool = ThreadPool(12)
vertex_vector = np.zeros(vertex_num, dtype=np.int32)

def f1_1(i):
    print ("Processing /out_convert/part-"+str(i).zfill(5))
    # cat = subprocess.Popen(["/opt/hadoop-1.2.1/bin/hadoop", "fs", "-cat", "/out_convert/part-"+str(i).zfill(5)], stdout=subprocess.PIPE)
    subprocess.Popen(["/opt/hadoop-1.2.1/bin/hadoop", "fs", "-get", "/out_convert/part-"+str(i).zfill(5), tmp_path+"/part-"+str(i).zfill(5)], stdout=subprocess.PIPE)
    a = []
    b = []
    cat = open(tmp_path+"/part-"+str(i).zfill(5), 'r')
    for line in cat.stdout:
        data = re.findall(r"[\w']+", line)
        a.append(int(data[0]))
        b.append(int(data[1]))
    cat.close()
    return a, b

results = pool.map(f1_1, range(100))
for i in results:
    vertex_vector[i[0]] = i[1]
    
edge_num = vertex_vector.sum()
print ("Edge Number: " + str(edge_num))
edge_per_partition = int(math.ceil(edge_num/partition_num))
print ("Edges Per Partition: " + str(edge_per_partition))
vertex_vector = np.cumsum(vertex_vector)
vertex_vector = np.append(0, vertex_vector)
partitioner = np.zeros(partition_num, dtype=np.int32)

def f1_2(i):
    return np.argmax(vertex_vector>=edge_per_partition*i)

results = pool.map(f1_2, range(1, partition_num))
partitioner[0:-1] = results
partitioner[partition_num-1] = vertex_num
del vertex_vector
print (str(partitioner))

broadcastVar = sc.broadcast(partitioner)
partitioner_t = broadcastVar.value
partitioner = np.append(0, partitioner)

def f2(line):
    a,b = line.split()
    return np.argmax(partitioner_t>int(b)), (int(b), int(a))
mat = lines.map(f2)
mat.persist(StorageLevel.MEMORY_AND_DISK)

partition_per_round = int(math.ceil(partition_num/round_num))
for i in range(round_num):
    start_partition = partition_per_round * i
    end_partition = min(partition_per_round * (i+1), partition_num)
    start_partition_b = sc.broadcast(start_partition)
    end_partition_b = sc.broadcast(end_partition)
    start_partition = start_partition_b.value
    end_partition = end_partition_b.value
    def f2_1(line):    
        partition_id = line[0]
        if partition_id >= start_partition and partition_id < end_partition:
            return True
        else:
            return False
    def f2_2(line):
        return line[0], line[1]
        #a,b = line.split()
        #return np.argmax(partitioner_t>int(b)), (int(b), int(a))
    mat_group = mat.filter(f2_1).groupByKey(numPartitions=partition_per_round).mapValues(list)
    def f3(list_data):
        edge_data = np.array(list_data[1])
        partition_id = list_data[0]
        start_id = partitioner[list_data[0]]
        end_id = partitioner[list_data[0]+1]
        edge_num = len(list_data[1])
        return partition_id, start_id, end_id, edge_num, edge_data.min(0)[0], edge_data.max(0)[0]
    tmp = mat_group.map(f3)
    tmp2 = tmp.collect()
    for i in tmp2:
        print ('########################## ' + str(i));

'''
'''
'''
def f4(list_data):
    edge_data = np.array(list_data[1])
    partition_id = list_data[0]
    start_id = partitioner[list_data[0]]
    end_id = partitioner[list_data[0]+1]
    edge_num = len(list_data[1])
    M = end_id - start_id
    N = vertex_num
    data = np.ones(edge_num, dtype=np.bool)
    p_mat = sparse.csr_matrix((data, (edge_data[:,0]-start_id, edge_data[:,1])), shape=(M, N))
    flushed_data = np.append(edge_num, len(p_mat.indices))
    flushed_data = np.append(flushed_data, len(p_mat.indptr))
    flushed_data = np.append(flushed_data, start_id)
    flushed_data = np.append(flushed_data, end_id)
    flushed_data = np.append(flushed_data, p_mat.indices)
    flushed_data = np.append(flushed_data, p_mat.indptr)
    flushed_data = flushed_data.astype(np.int32)
    # _file = open("/home/mapred/share/tmp/" + str(partition_id) + '.edge', 'w')
    # flushed_data.tofile(_file)
    # _file.close()
    return partition_id, start_id, end_id, len(flushed_data)
# mat.foreach(f4)
tmp = mat.map(f4)
tmp2 = tmp.collect()
for i in tmp2:
    print i;
'''
subprocess.Popen(["/opt/hadoop-1.2.1/bin/hadoop", "fs", "-rmr", "/out_convert"], stdout=subprocess.PIPE)
subprocess.Popen(["rm", tmp_path+"part-*"], stdout=subprocess.PIPE)
sc.stop()
