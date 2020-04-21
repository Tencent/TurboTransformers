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

set -x
if [ ! -n "$PREFIX" ]; then
  PREFIX="$CONDA_PREFIX"
fi
if [ ! -n "$PYTHON" ]; then
  PYTHON=`which python`
fi

export $(cat ./conda/.envs| xargs)


rm -rf build || true

mkdir -p build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release  ..
ninja

${PYTHON} -m pip install --no-deps --ignore-installed $(find . -name '*.whl')
