## turbo_transformers: a fast and user-friendly runtime for transformer inference on CPU and GPU
![logo](./images/logo.jpeg)

<center>Make transformers serving fast by adding a turbo to your inference engine!</center>

Transformer is the most critical alogrithm innovation in the NLP field in recent years. It brings higher model accuracy while introduces more calculations. The efficient deployment of online Transformer-based services faces enormous challenges. In order to make the costly Transformer online service more efficient, the WeChat AI open-sourced a Transformer inference acceleration tool called TurboTransformers, which has the following characteristics.
1. Supporting both Transformers Encoder and Decoder.
2. Excellent CPU / GPU performance. For Intel multi-core CPU and NVIDIA GPU hardware platforms, TurboTransformers can fully utilize all levels of computing power of the hardware. It has achieved better performance over pytorch / tensorflow and current mainstream optimization engines (such as TensorRT, torch JIT, NVIDIA faster transformers) on a variety of CPU and GPU hardware. See the detailed benchmark results below.
3. Tailored to the characteristics of NLP inference tasks. Unlike the CV task, the input dimensions of the NLP inference task always change. The traditional approach is zero padding or truncation to a fixed length, which introduces additional zero padding computational overhead. Besides, some frameworks such as onnxruntime, tensorRT, and torchlib need to preprocess the compuatation-graph according to the input size in advance for the best performance, which is not suitable for NLP tasks with varying sizes. TurboTransformers can support variable-length input sequence processing without preprocessing.
4. A simpler method of use. TurboTransformers supports python and C++ interface for calling. It can be used as an acceleration plug-in for pytorch. In the Transformer task, the end-to-end acceleration effect obtained by adding a few lines of python code.

TurboTransformers has been applied to multiple online BERT service scenarios in Tencent.
For example, It brings 1.88x acceleration to the WeChat FAQ service, 2.11x acceleration to the public cloud sentiment analysis service, and 13.6x acceleration to the QQ recommendation system.
Moreover, it have already been applied to build services such as Chitchating, Searching and Recommandation.

The following table is a comparison of TurboTransformers and related work.

| Related Works  |  Performance | Need Preprocess  |  Variable Length  | Usage |
|------------------|---|---|---|---|
| pytorch JIT (CPU) |  Fast |  Yes  | No  | Hard   |
| TensorRT (GPU) | Fast | Yes  | No  | Hard  |
| tf-Faster Transformers (GPU) | Fast  | Yes  | No  | Hard  |
| ONNX-runtime (CPU/GPU) | Fastest/Fast | No  | No  | Easy  |
| tensorflow-1.x (CPU/GPU) | Slow/Medium | Yes | No | Easy |
| pytorch (CPU/GPU) | Medium/Medium | No | Yes | Easy |
| **turbo-transformers (CPU/GPU)** | **Fastest/Fastest** | **No** | **Yes** | **Easy** |

### Supported Models
We currenly support the following transformer models.

* [BERT](https://arxiv.org/abs/1810.04805) [[Python]](./example/python/bert_example.py) [[C++]](./example/python/bert_example.cpp)
* [ALBERT](https://arxiv.org/abs/1909.11942) [[Python]](./example/python/albert_example.py)
* [Roberta](https://arxiv.org/abs/1907.11692) [[Python]](./example/python/roberta_example.py)
* [Transformer Decoder](https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py) [[Python]](https://github.com/TurboNLP/Translate-Demo)
* [GPT2](https://www.ceid.upatras.gr/webpages/faculty/zaro/teaching/alg-ds/PRESENTATIONS/PAPERS/2019-Radford-et-al_Language-Models-Are-Unsupervised-Multitask-%20Learners.pdf) [[Python]](./example/python/gpt2_example.py)

### Boost BERT Inference in 2 Lines of Python Code
```python
import torch
import transformers
import turbo_transformers

if __name__ == "__main__":
    turbo_transformers.set_num_threads(4)
    torch.set_num_threads(4)
    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(model_id)
    model.eval()
    cfg = model.config

    input_ids = torch.tensor(
        ([12166, 10699, 16752, 4454], [5342, 16471, 817, 16022]),
        dtype=torch.long)
    position_ids = torch.tensor(([1, 0, 0, 0], [1, 1, 1, 0]), dtype=torch.long)
    segment_ids = torch.tensor(([1, 1, 1, 0], [1, 0, 0, 0]), dtype=torch.long)
    torch.set_grad_enabled(False)
    torch_res = model(
        input_ids, position_ids=position_ids, token_type_ids=segment_ids
    )  # sequence_output, pooled_output, (hidden_states), (attentions)
    torch_seqence_output = torch_res[0][:, 0, :]
    tt_model = turbo_transformers.BertModel.from_torch(model)
    res = tt_model(
        input_ids, position_ids=position_ids,
        token_type_ids=segment_ids)  # pooled_output, sequence_output
    tt_seqence_output = res[0]
```

### Installation
Note that the building scripts only applie to specific OS and software (Pytorch, OpenNMT, transformers, etc.) versions.
Please adjust them according to your needs.

#### CPU
```
git clone https://github.com/Tencent/TurboTransformers --recursive
```
1. build docker images and containers on your machine.
```
sh tools/build_docker_cpu.sh
# optional: If you want to compare the performance of onnxrt-mkldnn during benchmark, you need to set BUILD_TYPE=dev to compile onnxruntime into the docker image, as follows
env BUILD_TYPE=dev sh tools/build_docker_cpu.sh
docker run -it --rm --name=turbort -v $PWD:/workspace your_image_name /bin/bash
```

2. Install turbo in docker

Method 1: I want to unitest
```
cd /workspace
sh tools/build_and_run_unittests.sh $PWD -DWITH_GPU=OFF
# you can switch between Openblas and MKL by modifying this line in CMakeList.txt
# set(BLAS_PROVIDER "mkl" CACHE STRING "Set the blas provider library, in [openblas, mkl, blis]")

```
Method 2: I do not want to unitest
```
cd /workspace
mkdir -p build && cd build
cmake .. -DWITH_GPU=OFF
make -j 4
pip install `find . -name *whl`
```
3. Run benchmark (optional) in docker, compare with pytorch, torch-JIT, onnxruntime
```
cd benchmark
bash run_benchmark.sh
```
4. Install conda packages in docker (optional)
```
sh tool/build_conda_package.sh
# The conda package will be in /workspace/dist/*.tar.bz2
# When using turbo_transformers in other environments outside this container: conda install your_root_path/dist/*.tar.bz2
```

*We also prepared a docker image containing CPU version of TurboTransformers, as well as other related works, i.e. onnxrt v1.2.0 and pytorch-jit on dockerhub*
```
docker pull thufeifeibear/turbo_transformers_cpu:latest
```
#### GPU
```
git clone https://github.com/Tencent/TurboTransformers --recursive
```
1. build docker images and containers on your machine.
```
# You can modify the environment variables in the script to specify the cuda version and operating system version
sh tools/build_docker_gpu.sh $PWD
nvidia-docker run --gpus all --net=host --rm -it -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=your_container_name REPOSITORY:TAG
# for example: nvidia-docker run --gpus all --net=host --rm -it -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=turbo_gpu_env thufeifeibear:0.1.1-cuda9.0-ubuntu16.04-gpu-dev
```

2. Install pip package in docker and unitest test
```
cd /workspace
sh tools/build_and_run_unittests.sh $PWD -DWITH_GPU=ON
```

3. Run benchmark (optional) in docker container, compare with pytorch
```
cd benchmark
bash gpu_run_benchmark.sh
```
We also prepared a docker image containing GPU version of TurboTransformers.
```
docker pull thufeifeibear/turbo_transformers_gpu:latest
```

### Usage
TurboTransformers provides C++ / python API interfaces. We hope to do our best to adapt to a variety of online environments to reduce the difficulty of development for users.

The first step in using turbo is to load a pre-trained model. We provide a way to load pytorch and tensorflow pre-trained models in [huggingface/transformers](https://github.com/huggingface).
The specific conversion method is to use the corresponding script in ./tools to convert the pre-trained model into an npz format file, and turbo uses the C ++ or python interface to load the npz format model.
In particular, we consider that most of the pre-trained models are in pytorch format and used with python. We provide a shortcut for calling directly in python for the pytorch saved model.

<img width="700" height="150" src="./images/pretrainmodelload.jpg" alt="pretrained">

### Examples
##### python APIs
Refer to examples of supported models in [./example/python](./example/python "python").
[TurboNLP/Translate-Demo](https://github.com/TurboNLP/Translate-Demo "translate") shows a demo of applying TurboTransformer in Translatetion Task.
Since the user of BERT acceleration always requires a customized post-processing process for the task, we provide an example of how to write a sequence classification application.
##### C++ APIs
Refer to [./example/cpp](./example/cpp "C ++") for an example.
Our example provides the GPU and two CPU multi-thread calling methods. One is to do one BERT inference using multiple threads; the other is to do multiple BERT inference, each of which using one thread.
Users can link turbo-transformers to your code through add_subdirectory.

## Performance
[BERT Benchmark Results](./docs/bert.md)

[ALBERT Benchmark Results](./docs/albert.md)

[Transformer Docoder Results](./docs/decoder.md)


## How to contribute new models
[How to know hotspots of your code?](./docs/profiler.md)

[How to add a new layer?](./turbo_transformers/layers/README.md)


## TODO
Currently (June 2020), In the near futuer, we will add support for low-precision models (CPU int8, GPU FP16).
**Looking forwards to your contribution!**

## Lisence
BSD 3-Clause License

## Known Issues
1. The results of Turbo Transformers may be different from the results of PyTorch after 2 digits behind the decimal point.
The diff mainly comes from Bert Output Layer. We use a approximate GELU algorithm, which may be different from PyTorch.
2. Turbo and PyTorch share the same MKL. MKL of PyTorch 1.5.0 may slow in Turbo. Reasons needs to be determined.
Download PyTorch version to 1.1.0 will improve Turbo's Performance.

## History
1. July 2020 v0.4.0, TurboTransformers used onnxruntime as cpu backend, supports GPT2. Anded a Quantized BERT.
2. July 2020 v0.3.1, TurboTransformers added support for ALbert, Roberta on CPU/GPU.
3. June 2020 v0.3.0, TurboTransformers added support for Transformer Decoder on CPU/GPU.
4. June 2020 v0.2.1, TurboTransformers added BLIS as a BLAS provider option. Better performance on AMD CPU.
5. April 2020 v0.0.1, TurboTransformers released, and achieved state-of-the-art BERT inference speed on CPU/GPU.


## Contact us
Although we recommand you post your problem with github issues, you can also join in our Turbo user group.
1. Scan this [QR code](./images/namecode.pdf "qrcode") and add our contactor as your WeChat friend.
2. QQ Group, Name: TurboTransformers, Number : 1109315167.
