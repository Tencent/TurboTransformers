#!/bin/bash
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

set -xe

SRC_ROOT=$1
BUILD_PATH=/tmp/build_cpu
bash ${SRC_ROOT}/tools/compile.sh ${SRC_ROOT} -DWITH_GPU=OFF ${BUILD_PATH}
python3 -m pip install -r ${SRC_ROOT}/requirements.txt
cd ${BUILD_PATH}
ctest --output-on-failure
# test npz model loader
# python ${SRC_ROOT}/tools/convert_huggingface_bert_pytorch_to_npz.py bert-base-uncased bert_torch.npz
# python ${SRC_ROOT}/example/python/bert_example.py bert_torch.npz
# rm bert_torch.npz
# pip install tensorflow
# python ${SRC_ROOT}/tools/convert_huggingface_bert_tf_to_npz.py bert-base-uncased bert_tf.npz
# python ${SRC_ROOT}/example/python/bert_example.py bert_tf.npz
# rm bert_tf.npz

BUILD_PATH=/tmp/build_gpu
bash ${SRC_ROOT}/tools/compile.sh ${SRC_ROOT} -DWITH_GPU=ON $BUILD_PATH
