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

CUDA_VERSION=10.1
PYTORCH_VERSION=1.7.0
BUILD_TYPES=("dev" "release")
EXTRA_ARGS="--build-arg https_proxy=http://192.168.12.11:1080"

DEV_IMAGE=thufeifeibear/turbo_transformers_gpu_dev:latest

for BUILD_TYPE in ${BUILD_TYPES[*]}
do
  if [ $BUILD_TYPE == "dev" ]; then
      NV_BASE_IMAGE=${CUDA_VERSION}-devel-ubuntu18.04
  elif [ $BUILD_TYPE == "release" ]; then
      NV_BASE_IMAGE=${CUDA_VERSION}-runtime-ubuntu18.04
  fi

  sed 's#IMAGE_BASE#nvidia/cuda:'${NV_BASE_IMAGE}'#g' ./docker/Dockerfile_${BUILD_TYPE}.gpu |
  sed 's#CUDA_VERSION#'${CUDA_VERSION}'#g'         |
  sed 's#PYTORCH_VERSION#'${PYTORCH_VERSION}'#g'   |
  sed 's#DEV_IMAGE#'${DEV_IMAGE}'#g'               > Dockerfile_${BUILD_TYPE}.gpu

  docker build ${EXTRA_ARGS} -t thufeifeibear/turbo_transformers_gpu_${BUILD_TYPE}:latest \
    -t thufeifeibear/turbo_transformers:${VERSION}-cuda${CUDA_VERSION}-gpu-${BUILD_TYPE} -f Dockerfile_${BUILD_TYPE}.gpu  .

done