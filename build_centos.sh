# export PATH=$HOME/opt/miniconda3/bin:${PATH}
# export CONDA_PREFIX=$HOME/opt/miniconda3
# curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh
# bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -p $HOME/opt/miniconda3 -b
# conda install pytorch=1.7.0 cudatoolkit=10.1 cudnn -c pytorch -y
# conda install conda-verify conda-build mkl-include cmake ninja -c anaconda -y
# pip install --no-cache-dir OpenNMT-py==1.1.0 docopt onnxruntime-gpu==1.3.0
# pip install -r requirements.txt
mkdir -p build && cd build && cmake .. -DWITH_GPU=ON && make -j
pip install `find . -name *whl`
