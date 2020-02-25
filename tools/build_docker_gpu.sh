#!/bin/bash
cd $(dirname $0)/../
set -xe

SOTWARE_VERSION=$(cat CMakeLists.txt | grep FAST_TRANSFORMERS_VERSION | \
    sed 's#set(FAST_TRANSFORMERS_VERSION ##g' | sed 's#)##g')
CUDA_VERSION=10.0
LINUX_VERSION=ubuntu18.04

docker build ${EXTRA_ARGS} \
	--build-arg CUDA_VERSION=${CUDA_VERSION} --build-arg LINUX_VERSION=${LINUX_VERSION} -t ccr.ccs.tencentyun.com/mmspr/fast_transformer:${SOTWARE_VERSION}-cuda${CUDA_VERSION}-gpu-dev -f Dockerfile.gpu_dev  .
