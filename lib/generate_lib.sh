gcc -c -O2 -fPIC satgraph.c
gcc -shared satgraph.o -o libsatgraph.so
rm satgraph.o
