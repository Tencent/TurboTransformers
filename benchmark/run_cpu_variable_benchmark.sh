# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

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
