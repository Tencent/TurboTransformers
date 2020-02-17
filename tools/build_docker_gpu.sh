#!/bin/bash
cd $(dirname $0)/../
set -xe

VERSION=$(cat CMakeLists.txt | grep FAST_TRANSFORMERS_VERSION | \
    sed 's#set(FAST_TRANSFORMERS_VERSION ##g' | sed 's#)##g')

docker build ${EXTRA_ARGS} \
	-t ccr.ccs.tencentyun.com/mmspr/fast_transformer:${VERSION}-gpu-dev -f Dockerfile.gpu_dev  .
