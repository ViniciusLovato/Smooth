#!/bin/bash
# ./run.sh <input_file>


mkdir -p output

# Run 10 times the sequential algoritm
for i in `seq 1 10`
	do
	    echo Executing sequential ${1}.txt iteration ${i} of 10
	    (/usr/bin/time -f%e bin/smooth data/${1}.txt) 2>> output/seq_${1}.out
	done

for j in `seq 1 10`
	do
        echo Executing parallel ${1}.txt iteration ${j} of 10
        (/usr/bin/time -f%e bin/parallelSmooth data/${1}.txt) 2>> output/par_${1}.out
    done
    
done
