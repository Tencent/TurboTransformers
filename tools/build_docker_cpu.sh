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

cd $(dirname $0)
set -xe

VERSION=$(cat ../CMakeLists.txt | grep turbo_transformers_VERSION | \
    sed 's#set(turbo_transformers_VERSION ##g' | sed 's#)##g')

BUILD_TYPE=release

docker build ${EXTRA_ARGS} \
  -t ccr.ccs.tencentyun.com/mmspr/turbo_transformers:${VERSION}-${BUILD_TYPE} -f ./docker/Dockerfile_${BUILD_TYPE}.cpu .
