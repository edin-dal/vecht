#!/bin/bash
###################################################################################################################
printf '### Running End-to-End Benchmarks...\n\n'
echo off > /sys/devices/system/cpu/smt/control
mkdir -p results
chmod 777 results
echo "" > results/e2e_query.txt
echo "" > results/e2e_other.csv
mkdir -p results/micro
chmod 777 results/micro
###################################################################################################################
printf '1) End-to-End Benchmarks (Query)...\n'
printf 'Saving results into results/e2e_query.txt\n\n'
if [ -z "$TPCH_PATH" ]
then
    echo "TPCH_PATH is not set. Please set it to the path of the TPCH data files."
    exit 1
fi
./e2e_query.o $TPCH_PATH >> results/e2e_query.txt
###################################################################################################################
printf '2) End-to-End Benchmarks (Other)...\n'
printf 'Saving results into results/e2e_other.txt\n\n'
size=16777216
fixedsel=0.015625
maxval=100
threads=(1 4)
selectivity=(0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5)
firstcall=1
for threads in "${threads[@]}"
do
    for sel in "${selectivity[@]}"
    do
        echo "./e2e_other.o $size $size $fixedsel $sel $maxval $threads $firstcall >> results/e2e_other.csv"
        ./e2e_other.o $size $size $fixedsel $sel $maxval $threads $firstcall >> results/e2e_other.csv
        echo "done."
        firstcall=0
    done
done
###################################################################################################################
printf '### Running Micro-Benchmarks...\n'
printf 'Saving results into results/micro/\n\n'
outer_size=1500000
threads=(1 4)
selectivities=(0.1 0.5 1)
for threads in "${threads[@]}"
do
    for selectivity in "${selectivities[@]}"
    do
        firstcall=1
        echo "" > results/micro/res-$threads-$outer_size-$selectivity.csv
        for log_table_bytes in {20..29}
        do
            echo "./micro.o $log_table_bytes $outer_size $selectivity 0.5 $threads $firstcall >> results/micro/res-$threads-$outer_size-$selectivity.csv"
            ./micro.o $log_table_bytes $outer_size $selectivity 0.5 $threads $firstcall >> results/micro/res-$threads-$outer_size-$selectivity.csv
            echo "done."
            firstcall=0
        done
    done
done
###################################################################################################################
printf '### Running Done.\n\n'
