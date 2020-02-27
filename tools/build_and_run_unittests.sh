#!/bin/bash
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
