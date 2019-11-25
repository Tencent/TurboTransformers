#!/bin/bash
cd $(dirname $0)/../
set -xe
docker build ${EXTRA_ARGS} \
	-t ccr.ccs.tencentyun.com/mmspr/fast_transformer:0.1.1-dev -f Dockerfile.dev  .
