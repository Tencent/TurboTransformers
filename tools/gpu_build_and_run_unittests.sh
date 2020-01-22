#!/bin/bash

# This script is made for the image chengduozh/ft_gpu_dev:0.3.0
# on mnet gpu4 in Tencent IDC cluster.
set -xe
SRC_ROOT=$1

mkdir -p /myspace/.hunter
tar -xf  /workspace/base.tar.gz  -C /myspace/.hunter
export HUNTER_ROOT=/myspace/.hunter

export BUILD_ROOT=/myspace/build
rm -rf ${BUILD_ROOT} || true
mkdir -p ${BUILD_ROOT}
cd ${BUILD_ROOT}
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON ${SRC_ROOT}
make -j ${nproc}
ctest --output-on-failure
