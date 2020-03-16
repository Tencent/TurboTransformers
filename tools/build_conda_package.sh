#!/usr/bin/env bash
cd $(dirname $0)
cd ./conda
env | grep proxy > .envs
conda-build .
rm .envs
mkdir -p ../../dist
cp /opt/conda/conda-bld/linux-64/easy-transformers-*.tar.bz2 ../../dist/
