#!/bin/bash


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
