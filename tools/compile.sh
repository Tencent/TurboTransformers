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

set -xe
SRC_ROOT=$1
WITH_GPU=$2

BUILD_PATH=/tmp/build
if [ $# == 3 ] ; then
    BUILD_PATH=$3
fi

rm -rf ${BUILD_PATH} || true
mkdir -p ${BUILD_PATH}
cd ${BUILD_PATH}
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release ${SRC_ROOT} ${WITH_GPU}
ninja
pip install `find . -name "*whl"`
