#/opt/openmpi/bin/mpirun  -n 18  -hostfile  ../hostfile2 -use-hwthread-cpus --map-by node -cpus-per-proc 12  /opt/intel/intelpython27/bin/python satgraph_tq_2.py
#/opt/openmpi/bin/mpirun  -n 18  -hostfile  ../hostfile2 -use-hwthread-cpus --map-by node -cpus-per-proc 12  /opt/intel/intelpython27/bin/python satgraph_tq_3.py
/opt/openmpi/bin/mpirun  -n 9  -hostfile  ../hostfile2 -use-hwthread-cpus --map-by node -cpus-per-proc 24  /opt/intel/intelpython27/bin/python satgraph_tq_3.py
