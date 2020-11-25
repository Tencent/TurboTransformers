#!/bin/bash


cd $(dirname $0)/
set -xe
VERSION=$(cat ../CMakeLists.txt | grep TURBO_TRANSFORMERS_VERSION | \
    sed 's#set(TURBO_TRANSFORMERS_VERSION ##g' | sed 's#)##g')

CUDA_VERSION=10.1
PYTORCH_VERSION=1.7.0
BUILD_TYPES=("dev")

DEV_IMAGE=ppopp21whoami/turbo_transformers_gpu_dev:latest

for BUILD_TYPE in ${BUILD_TYPES[*]}
do
  if [ $BUILD_TYPE == "dev" ]; then
      NV_BASE_IMAGE=${CUDA_VERSION}-devel-ubuntu18.04
  elif [ $BUILD_TYPE == "release" ]; then
      NV_BASE_IMAGE=${CUDA_VERSION}-base-ubuntu18.04
  fi

  sed 's#IMAGE_BASE#nvidia/cuda:'${NV_BASE_IMAGE}'#g' ./docker/Dockerfile_${BUILD_TYPE}.gpu |
  sed 's#CUDA_VERSION#'${CUDA_VERSION}'#g'         |
  sed 's#PYTORCH_VERSION#'${PYTORCH_VERSION}'#g'   |
  sed 's#DEV_IMAGE#'${DEV_IMAGE}'#g'               > Dockerfile_${BUILD_TYPE}.gpu

  docker build ${EXTRA_ARGS} -t ppopp21whoami/turbo_transformers_gpu_${BUILD_TYPE}:latest \
    -t ppopp21whoami/turbo_transformers:${VERSION}-cuda${CUDA_VERSION}-gpu-${BUILD_TYPE} -f Dockerfile_${BUILD_TYPE}.gpu  .

done
