#!/bin/bash
set -xe
SRC_ROOT=$1
rm -rf /tmp/build || true
mkdir -p /tmp/build
cd /tmp/build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release ${SRC_ROOT} -DWITH_GPU=ON
ninja
pip install `find . -name "*whl"`
python3 -m pip install contexttimer
ctest --output-on-failure
