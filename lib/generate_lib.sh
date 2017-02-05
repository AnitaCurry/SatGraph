gcc -c -O3 -fPIC -fopenmp -Wall  satgraph.c
#gcc -c -fPIC -Wall  satgraph.c
gcc -shared -fopenmp  satgraph.o -o libsatgraph.so
#gcc -shared   satgraph.o -o libsatgraph.so
rm satgraph.o
