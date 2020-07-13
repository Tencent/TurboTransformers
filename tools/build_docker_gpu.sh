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

cd $(dirname $0)/
set -xe
VERSION=$(cat ../CMakeLists.txt | grep TURBO_TRANSFORMERS_VERSION | \
    sed 's#set(TURBO_TRANSFORMERS_VERSION ##g' | sed 's#)##g')

if [ -z $BUILD_TYPE ]; then
  BUILD_TYPE=release
fi

CUDA_VERSION=10.1
DOCKER_BASE=${CUDA_VERSION}-cudnn7-devel-ubuntu18.04
PYTORCH_VERSION=1.5.0
sed 's#IMAGE_BASE#nvidia/cuda:'${DOCKER_BASE}'#g' ./docker/Dockerfile_${BUILD_TYPE}.gpu |
sed 's#CUDA_VERSION#'${CUDA_VERSION}'#g'         |
sed 's#PYTORCH_VERSION#'${PYTORCH_VERSION}'#g'    > Dockerfile.gpu

docker build ${EXTRA_ARGS} \
  -t thufeifeibear/turbo_transformers:${VERSION}-cuda${DOCKER_BASE}-gpu-${BUILD_TYPE} -f Dockerfile.gpu  .
