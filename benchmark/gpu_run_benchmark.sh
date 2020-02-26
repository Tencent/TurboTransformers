#!/bin/bash
set -e
FRAMEWORKS=("fast-transformers" "torch")
SEQ_LEN=(10 20 40 60 80 120, 200, 300, 400, 500)
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
          -n ${N} --framework=${framework}
    done
  done
done
