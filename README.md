## TurboTransformer User Guidance
### Installation
1. build docker images and containers on your machine.
```
bash tools/build_docker_gpu.sh $PWD
nvidia-docker run --gpus all --net=host --rm -it -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=turbo_gpu_env ppopp21whoami/turbo_transformers_gpu_dev:latest
```

2. Install pip package in docker container and run unitests
```
# In the docker container turbo_gpu_env
cd /workspace
bash tools/build_and_run_unittests.sh $PWD -DWITH_GPU=ON
```

**Note** that you can skip Step 1, 2 by using a pre-built docker image `ppopp21whoami/turbo_transformers_gpu_dev:latest` on dockerhub (based on CUDA 10.1, ubuntu 18.04).
```
docker pull ppopp21whoami/turbo_transformers_gpu_dev:latest
nvidia-docker run --gpus all --net=host --rm -it -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=turbo_gpu_env ppopp21whoami/turbo_transformers_gpu_dev:latest
```

### Reproduce the results in the manuscript
3. Run benchmark in docker container, compare with pytorch
```
# In the docker container turbo_gpu_env
cd benchmark
# Test Efficiency on variable-length input for Bert and Albert
bash gpu_run_variable_benchmark.sh
# Test Efficiency on fixed-length input for Bert and Albert
bash gpu_run_variable_benchmark.sh > fixed_length_res.txt
```
These above two script will reproduce the BERT and ALBERT results in Figure 10 and Figure 11 of the manuscript.
The gpu_run_variable_benchmark.sh will produce the following 4 files.

torch_4_albert_latency.txt
torch_4_bert_latency.txt
turbo_4_albert_latency.txt
turbo_4_bert_latency.txt

The gpu_run_variable_benchmark.sh will output results in format of json to the file fixed_length_res.txt.

### Claims
1. The artifact provides the necessary code and data to reproduce the performance results of the runtime on a base BERT model and a large Albert.
2. The artifact dose not provides the data to reproduce the performance results on a transformer decoder,
because the neural translation model used in the paper does not have an open source permit right now.
However, users can easily apply the runtime to their own decoder models.
3. The artifact dose not provides the code to reproduce the serving framework results.
Because the serving framework used in the paper does not have an open source permit right now.
