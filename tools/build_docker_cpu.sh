#!/bin/bash
cd $(dirname $0)
set -xe

VERSION=$(cat ../CMakeLists.txt | grep turbo_transformers_VERSION | \
    sed 's#set(turbo_transformers_VERSION ##g' | sed 's#)##g')

docker build ${EXTRA_ARGS} \
	-t ccr.ccs.tencentyun.com/mmspr/turbo_transformers:${VERSION}-dev -f ./docker/Dockerfile_dev.cpu .
