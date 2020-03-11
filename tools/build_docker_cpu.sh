#!/bin/bash
cd $(dirname $0)/../
set -xe

VERSION=$(cat CMakeLists.txt | grep easy_transformers_VERSION | \
    sed 's#set(easy_transformers_VERSION ##g' | sed 's#)##g')

docker build ${EXTRA_ARGS} \
	-t ccr.ccs.tencentyun.com/mmspr/easy_transformers:${VERSION}-dev -f ./docker/Dockerfile_dev.cpu .
