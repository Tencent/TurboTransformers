#!/bin/bash
set -e
#NUM_THREADS=(4)
FRAMEWORKS=("fast-transformers")
#BATCH_SIZE=(2)
#SEQ_LEN=(4)
NUM_THREADS=(1 2 4 8)
#FRAMEWORKS=("torch" "torch_jit" "fast-transformers" "onnxruntime-cpu" "onnxruntime-mkldnn")
SEQ_LEN=(10 20 40 60 80 120)
BATCH_SIZE=(1 2)
N=30
MODEL="bert-base-chinese"
for n_th in ${NUM_THREADS[*]}
do
  for batch_size in ${BATCH_SIZE[*]}
  do
    for seq_len in ${SEQ_LEN[*]}
    do
      for framework in ${FRAMEWORKS[*]}
      do
        python benchmark.py ${MODEL} --seq_len=${seq_len} --batch_size=${batch_size}\
            -n ${N} --framework=${framework} --num_threads=${n_th}
      done
    done
  done
done
