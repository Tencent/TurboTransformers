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

VERSION=$(cat ../CMakeLists.txt | grep turbo_transformers_VERSION | \
    sed 's#set(turbo_transformers_VERSION ##g' | sed 's#)##g')

if [ -z $BUILD_TYPE ]; then
  BUILD_TYPE=release
fi

docker build ${EXTRA_ARGS} \
  -t ccr.ccs.tencentyun.com/mmspr/turbo_transformers:${VERSION}-${BUILD_TYPE} -f ./docker/Dockerfile_${BUILD_TYPE}.cpu .
