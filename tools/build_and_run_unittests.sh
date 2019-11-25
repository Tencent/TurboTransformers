#!/bin/bash
set -xe
SRC_ROOT=$1
rm -rf /tmp/build || true
mkdir -p /tmp/build
cd /tmp/build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release ${SRC_ROOT}
ninja
python3 -m pip install -r ${SRC_ROOT}/test_requirements.txt
ctest --output-on-failure
