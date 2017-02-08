gcc -c -O3 -fPIC -fopenmp -Wall -I /opt/intel/intelpython27/include/python2.7  satgraph.cpp
#g++ -c -O2 -ftree-vectorize -msse2 -ftree-vectorizer-verbose=1 -fPIC -fopenmp -Wall satgraph.cpp

g++ -shared -fopenmp  satgraph.o -o libsatgraph.so
rm satgraph.o
