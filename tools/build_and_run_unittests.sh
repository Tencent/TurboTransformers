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
if [ $# != 2 ] ; then
  echo "USAGE: $0 PATH_TO_CODE WITH_GPU_OR_NOT"
  echo " e.g.: $0 `PWD` -DWITH_GPU=OFF"
  exit 1;
fi
SRC_ROOT=$1
WITH_GPU=$2
BUILD_PATH=/tmp/build_cpu
bash ${SRC_ROOT}/tools/compile.sh ${SRC_ROOT} ${WITH_GPU} ${BUILD_PATH}
python3 -m pip install -r ${SRC_ROOT}/requirements.txt
cd ${BUILD_PATH}
ctest --output-on-failure
