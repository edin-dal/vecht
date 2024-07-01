# !/bin/bash
printf '### Building Started...\n\n'
######################################################################################
printf '1) Building e2e_query.cpp\n\n'
g++ e2e_query.cpp -march=core-avx2 -O3 -ltbb -std=c++17 -o e2e_query.o 
######################################################################################
printf '2) Building e2e_other.cpp\n\n'
g++ e2e_other.cpp -march=core-avx2 -O3 -lpthread -lrt -ltbb -std=c++17 -o e2e_other.o
######################################################################################
printf '3) Building micro.cpp\n\n'
g++ micro.cpp ../include/inner_outer.c ../include/rand.c -march=core-avx2 -O3 -lpthread -lrt -ltbb -std=c++17 -o micro.o
######################################################################################
printf '### Building Done.\n\n'