#!/bin/bash
# ./run.sh <input_file>


mkdir -p output

# Run 10 times the sequential algoritm
for i in `seq 1 10`
	do
	    echo Executing sequential ${1}.jpg iteration ${i} of 10
	    (/usr/bin/time -f%e bin/smooth img/${1}.jpg) 2>> output/seq_${1}.out
	done

for j in `seq 1 10`
	do
        echo Executing parallel ${1}.jpg iteration ${j} of 10
        (/usr/bin/time -f%e mpirun -machinefile machinefile bin/parallelSmooth img/${1}.jpg) 2>> output/par_${1}.out
    done

for k in `seq 1 10`
	do
        echo Executing CUDA ${1}.jpg iteration ${j} of 10
        (/usr/bin/time -f%e bin/smoothCUDA img/${1}.jpg) 2>> output/CUDA_${1}.out
    done
    
