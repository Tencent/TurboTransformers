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
WITH_GPU=$2

BUILD_PATH=/tmp/build
if [ $# == 3 ] ; then
    BUILD_PATH=$3
fi

rm -rf ${BUILD_PATH} || true
mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}
cmake -G Ninja \
    -D CMAKE_BUILD_TYPE=Release -D CUDA_INCLUDE_DIRS=/usr/local/cuda/include ${SRC_ROOT} ${WITH_GPU}
ninja
pip install `find . -name "*whl"`
