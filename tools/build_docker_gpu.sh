#!/bin/bash
# Copyright 2020 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cd $(dirname $0)/
set -xe
VERSION=$(cat ../CMakeLists.txt | grep turbo_transformers_VERSION | \
    sed 's#set(turbo_transformers_VERSION ##g' | sed 's#)##g')

CUDA_VERSION=10.0
DOCKER_BASE=${CUDA_VERSION}-cudnn7-devel-ubuntu18.04
PYTORCH_VERSION=1.1.0
sed 's#IMAGE_BASE#nvidia/cuda:'${DOCKER_BASE}'#g' ./docker/Dockerfile_dev.gpu |
sed 's#CUDA_VERSION#'${CUDA_VERSION}'#g'         |
sed 's#PYTORCH_VERSION#'${PYTORCH_VERSION}'#g'    > Dockerfile.gpu

docker build ${EXTRA_ARGS} \
	-t ccr.ccs.tencentyun.com/mmspr/turbo_transformers:${VERSION}-cuda${DOCKER_BASE}-gpu-dev -f Dockerfile.gpu  .
