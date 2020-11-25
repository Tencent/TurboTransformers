#!/bin/bash


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
