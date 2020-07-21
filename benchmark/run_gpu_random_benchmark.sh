set -e
# Turbo is designed for variable-length input
# This script benchmarks turbo using a list of request with variable lengths
# FRAMEWORKS=("turbo-transformers" "torch" "onnxruntime")
# FRAMEWORKS=("turbo-transformers" "torch")
FRAMEWORKS=("onnxruntime")

MAX_SEQ_LEN=(10 20 40 60 80 100 200 300 400 500)

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
