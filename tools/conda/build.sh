#!/bin/bash


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
