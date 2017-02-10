gcc -c -O3 -ftree-vectorize -msse2 -ftree-vectorizer-verbose=1 -fPIC -fopenmp -Wall satgraph.c
#gcc -c -O3 -fPIC -fopenmp -Wall  satgraph.c

gcc -shared -fopenmp  satgraph.o -o libsatgraph.so
#gcc -shared   satgraph.o -o libsatgraph.so
rm satgraph.o
