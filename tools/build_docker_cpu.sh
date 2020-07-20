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

cd $(dirname $0)
set -xe

VERSION=$(cat ../CMakeLists.txt | grep TURBO_TRANSFORMERS_VERSION | \
    sed 's#set(TURBO_TRANSFORMERS_VERSION ##g' | sed 's#)##g')

if [ -z $BUILD_TYPE ]; then
  BUILD_TYPE=release
fi

EXTRA_BUILD_ARGS=""
EXTRA_RUN_ARGS=""
if [[ "x${http_proxy}" != "x" ]];then
  EXTRA_BUILD_ARGS="--build-arg http_proxy=${http_proxy}"
  EXTRA_RUN_ARGS="-e http_proxy=${http_proxy}"
fi

if [[ "x${https_proxy}" != "x" ]]; then
  EXTRA_BUILD_ARGS="${EXTRA_BUILD_ARGS} --build-arg https_proxy=${https_proxy}"
  EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS} -e https_proxy=${https_proxy}"
fi

docker build ${EXTRA_BUILD_ARGS} \
  -t thufeifeibear/turbo_transformers:${VERSION}-cpu-${BUILD_TYPE} -f ./docker/Dockerfile_${BUILD_TYPE}.cpu .
