#!/bin/bash
cd $(dirname $0)/
set -xe
VERSION=$(cat ../CMakeLists.txt | grep turbo_transformers_VERSION | \
    sed 's#set(turbo_transformers_VERSION ##g' | sed 's#)##g')

CUDA_VERSION=9.0
DOCKER_BASE=${CUDA_VERSION}-cudnn7-devel-ubuntu16.04
PYTORCH_VERSION=1.1.0
sed 's#IMAGE_BASE#nvidia/cuda:'${DOCKER_BASE}'#g' ./docker/Dockerfile_dev.gpu |
sed 's#CUDA_VERSION#'${CUDA_VERSION}'#g'         |
sed 's#PYTORCH_VERSION#'${PYTORCH_VERSION}'#g'    > Dockerfile.gpu

docker build ${EXTRA_ARGS} \
	-t ccr.ccs.tencentyun.com/mmspr/turbo_transformers:${VERSION}-cuda${DOCKER_BASE}-gpu-dev -f Dockerfile.gpu  .
