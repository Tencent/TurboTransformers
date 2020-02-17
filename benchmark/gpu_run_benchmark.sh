#!/bin/bash
set -e
#NUM_THREADS=(4)
#FRAMEWORKS=("fast-transformers")
#BATCH_SIZE=(2)
#SEQ_LEN=(4)
FRAMEWORKS=("fast-transformers" "torch")
SEQ_LEN=(10 20 40 60 80 120)
#SEQ_LEN=(500)
#SEQ_LEN=(150 200 250 300 350 400 450 500)
BATCH_SIZE=(1 20)
N=150
MODEL="bert-base-chinese"
for batch_size in ${BATCH_SIZE[*]}
do
  for seq_len in ${SEQ_LEN[*]}
  do
    for framework in ${FRAMEWORKS[*]}
    do
      python gpu_benchmark.py ${MODEL} --seq_len=${seq_len} --batch_size=${batch_size}\
          -n ${N} --framework=${framework} --num_threads=1
    done
  done
done
