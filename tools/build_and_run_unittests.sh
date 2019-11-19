#!/bin/bash
set -xe
SRC_ROOT=$1
mkdir -p /tmp/build
cd /tmp/build
cmake -DCMAKE_BUILD_TYPE=Release ${SRC_ROOT}
make VERBOSE=1 -j $(nproc)
pip3 install -r ${SRC_ROOT}/test_requirements.txt
ctest -j $(nproc) --output-on-failure
