#!/bin/bash


set -e
NUM_THREADS=(4 8)
FRAMEWORKS=("torch" "torch_jit" "turbo-transformers" "onnxruntime-cpu")
SEQ_LEN=(40 60 80 100 120 200 300 400 500)
BATCH_SIZE=(1 2)
N=150
MODEL="bert"
for n_th in ${NUM_THREADS[*]}
do
  for batch_size in ${BATCH_SIZE[*]}
  do
    for seq_len in ${SEQ_LEN[*]}
    do
      for framework in ${FRAMEWORKS[*]}
      do
        env OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=${n_th} python benchmark.py ${MODEL} --seq_len=${seq_len} --batch_size=${batch_size}\
            -n ${N} --framework=${framework} --num_threads=${n_th}
      done
    done
  done
done
