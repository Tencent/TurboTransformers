set -e
# Turbo is designed for variable-length input
# This script benchmarks turbo using a list of request with variable lengths
# FRAMEWORKS=("turbo-transformers" "torch" "onnxruntime")
FRAMEWORKS=("turbo-transformers" "torch")
# Note Onnx doese not supports Albert
# FRAMEWORKS=("onnxruntime")

MAX_SEQ_LEN=(500)

N=150
MODEL="albert"
for max_seq_len in ${MAX_SEQ_LEN[*]}
do
  for framework in ${FRAMEWORKS[*]}
  do
    python gpu_benchmark.py ${MODEL} \
              --enable-random \
              --min_seq_len=5  \
              --max_seq_len=${max_seq_len}  \
              --batch_size=1 \
              -n ${N} \
              --framework=${framework}
  done
done
