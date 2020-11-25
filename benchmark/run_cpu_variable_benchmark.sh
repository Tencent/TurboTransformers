

set -e
# Turbo is designed for variable-length input
# This script benchmarks turbo using a list of request with variable lengths
FRAMEWORKS=("turbo-transformers")
# FRAMEWORKS=("turbo-transformers" "torch")
# FRAMEWORKS=("torch")
# Note Onnx doese not supports Albert
# FRAMEWORKS=("onnxruntime-cpu")
NTHREADS=4
MAX_SEQ_LEN=(500)
N=150
MODEL="bert"
for max_seq_len in ${MAX_SEQ_LEN[*]}
do
  for framework in ${FRAMEWORKS[*]}
  do
    # env OMP_WAIT_POLICY=ACTIVE
    env OMP_NUM_THREADS=${NTHREADS} python benchmark.py ${MODEL} \
              --enable-random \
              --min_seq_len=5  \
              --max_seq_len=${max_seq_len}  \
              --batch_size=1 \
              -n ${N} \
              --num_threads=${NTHREADS} \
              --framework=${framework}
  done
done
