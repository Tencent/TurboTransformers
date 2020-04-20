## turbo_transformers: 面向CPU/GPU高效易用的Transformer推理工具
[English Version](./README.md)
![logo](./images/logo.jpeg)

### **为你的Transformer推理引擎使用涡轮增压吧!**

Transformer是近年来NLP领域最重要的模型创新，带来更高的模型精度的同时也引入了更多的计算量，线上Transformer服务的高效部署面临着巨大挑战。面对丰富的Transformer的线上服务场景，微信模式识别中心开源了名为TurboTransformers的Transformer推理加速引擎。Turbo具有如下特性。
1. 优异的CPU/GPU性能表现。面向Intel多核CPU和NVIDIA GPU硬件平台，TurboTransformers可以充发挥硬件的各层级计算能力。在多种CPU和GPU硬件上获得了超过pytorch/tensorflow和目前主流优化引擎（如onnxruntime-mkldnn/onnxruntime-gpu, torch JIT, NVIDIA faster transformers）的性能表现。详细benchmark结果见下文。
2. 为NLP推理任务特点量身定制。和CV任务不同，NLP推理任务输入尺寸多个维度会存在变化。传统做法是补零或者截断成固定长度，这样引入了额外补零计算开销。另外有些框架如onnxruntime、tensorRT、torchlib需要预先对计算图根据输入尺寸进行预处理，这对尺寸变化的NLP任务并不适用。TurboTransformers可以支持变长输入序列处理，且不需要预处理过程。
3. 更简单的使用方式。TurboTransformers支持python和C++接口进行调用。它可以作为pytorch的加速插件，在Transformer任务上，通过加入几行python代码获得的端对端加速效果。

TurboTransformers的已经应用于腾讯内部多个线上BERT服务服务场景。比如，微信的FAQ的服务获得1.88x加速，公有云情感分析服务获得2.11x加速，QQ推荐服务获得13.6x加速。

下表是本工作和相关工作的对比

| Related Works  |  Performance | Need Preprocess  |  Variable Length  | Usage |
|------------------|---|---|---|---|
| pytorch JIT (CPU) |  Fast |  Yes  | No  | Hard   |
| tf-Faster Transformers (GPU) | Fast  | Yes  | No  | Hard  |
| ONNX-runtime(CPU/GPU) | Fast/Fast | Yes  | No  | Easy  |
| tensorflow-1.x (CPU/GPU) | Slow/Medium | Yes | No | Easy |
| pytorch (CPU/GPU) | Medium/Medium | No | Yes | Easy |
| **turbo-transformers (CPU/GPU)** | **Fastest/Fastest** | **No** | **Yes** | **Easy** |


### CPU版本安装
#### 本机构建
recursive方式clone
1. 本机构建docker镜像和容器
```
sh tools/build_docker_cpu.sh
# optional: benchmark时如果想比较onnxrt-mkldnn的结果需要设置BUILD_TYPE=dev将onnxruntime编入docker镜像，如下
env BUILD_TYPE=dev sh tools/build_docker_cpu.sh
```

2. 在docker内进行安装
方法1：我想单测和benchmark
```
cd /workspace
sh tools/build_and_run_unittests.sh.sh $PWD -DWITH_GPU=OFF
```
方法2：我不想单测
```
cd /workspace
# 安装前需要跑一个单测，这里必须下载单测需要的预训练模型，需要git lfs，sudo yum install git-lfs
mkdir -p build && cd build
cmake .. -DWITH_GPU=OFF
pip install -r `find . -name *whl`
```
3. 在docker内运行benchmark (optional), 和pytorch, torch-JIT, onnxruntime比较
```
cd benchmark
bash run_benchmark.sh
```
4. 在docker内安装conda包（optional）

```
sh tool/build_conda_package.sh
# conda包会在 /workspace/dist/*.tar.bz2中
# 在本容器外其他环境使用turbo_transformers时只需要python -m pip install your_root_path/dist/*.tar.bz2
```

#### 使用腾讯云dockerhub镜像
前提：具有ccr.ccs.tencentyun.com/mmspr/turbo_transformers:0.1.1-dev权限
参考tools/docker/Dockerfile_tencentyun.template

### GPU版本安装
git clone https://git.code.oa.com/PRC_alg/fast_transformers --recursive
1. 本机构建docker镜像和容器
```
# 可以在脚本中修改环境变量指定cuda版本和操作系统版本
sh tools/build_docker_gpu.sh $PWD
docker run --net=host --rm -it -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=your_container_name REPOSITORY:TAG
# for example: docker run --net=host --rm -it -v $PWD:/workspace -v /etc/passwd:/etc/passwd --name=jiarui_gpu_env ccr.ccs.tencentyun.com/mmspr/turbo_transformers:0.1.1-cuda9.0-ubuntu16.04-gpu-dev
```

2. 在docker内安装pip包并单测
```
cd /workspace
sh tools/build_and_run_unittests.sh $PWD -DWITH_GPU=ON
```

3. 在docker内运行benchmark (optional), 和pytorch比较
```
cd benchmark
bash gpu_run_benchmark.sh
```

### 使用方法
turbo_transformers提供了简单的C++/python调用接口，我们希望尽最大努力适配多样的上线环境，减轻使用者的开发难度。

使用turbo的第一步是加载预训练好的模型，我们提供了载入[huggingface/transformers](https://github.com/huggingface)的pytorch和tensorflow预训练模型方式。
具体转换方式是使用tools的对应脚本，将预训练模型转换成npz格式的文件，turbo使用C++或者python接口载入npz格式模型。
特别的，我们考虑大部分预训练模型是pytorch格式的并使用python调用，我们针对pytorch saved模型提供了一个python方式直接调用的捷径。
<img width="700" height="150" src="./images/pretrainmodelload.jpg" alt="加载预训练模型">
#### python接口
参考[./example/python](./example/python "python")的例子。
由于使用BERT之后还需要针对任务定制的后处理过程，我们提供了一个bert sequence classification的书写方式示例。
在工蜂上我们还内部开源了一套可以使用turbo的python severing框架[bertserving](https://git.code.oa.com/PRC_alg/bert-serving/tree/develop "bertserving")供使用者参考，它通过asyncio方式异步响应BERT推理的http请求。
#### C++接口
参考[./example/cpp](./example/cpp "C++")的例子。
我们的例子提供了GPU和两种CPU多线程的调用方式。一种是串行响应BERT计算请求，每次BERT计算使用多线程（omp）方式计算，另一种是多线程并行的响应BERT计算请求，每次BERT计算使用单线程方式的方式。
用户使用时候可以通过add_subdirectory方式链接turbo-transformers。
## 性能
### CPU测试效果
我们在三种CPU硬件平台测试了TurboTransformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[pytorch-jit](https://pytorch.org/docs/stable/_modules/torch/jit.html "pytorch-jit")和[onnxruntime-mkldnn]( https://github.com/microsoft/onnxruntime "onnxruntime-mkldnn")和TensorRT实现作为对比。性能测试结果为迭代150次的均值。为了避免多次测试时，上次迭代的数据在cache中缓存的现象，每次测试采用随机数据，并在计算后刷新的cache数据。

* Intel Xeon 61xx

<img width="900" height="300" src="./images/61xx_perf_thd48_0415.jpg" alt="61xx性能">
<img width="900" height="300" src="./images/61xx_speedup_thd48_0415.jpg" alt="61xx加速">

* Intel Xeon 6133

相比61xx型号，Intel Xeon 6133向量化长度更长为512 bit，并且它拥有一个30 MB核间共享L3 cache。

<img width="900" height="300" src="./images/6133_perf_thd48_0415.jpg" alt="6133性能">
<img width="900" height="300" src="./images/6133_speedup_thd48_0415.jpg" alt="6133加速">

### GPU测试效果
我们在四种GPU硬件平台测试了turbo_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[NVIDIA Faster Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer "FasterTransformer")，[onnxruntime-gpu](https://github.com/microsoft/onnxruntime "onnxrt-gpu")[TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT)实现作为对比。性能测试结果为迭代150次的均值。

* RTX 2060
<img width="900" height="300" src="./images/2060-perf.jpg" alt="2060性能">
<img width="900" height="300" src="./images/2060-speedup.jpg" alt="2060加速">

* Tesla V100

<img width="900" height="300" src="./images/v100-perf.jpg" alt="V100性能">
<img width="900" height="300" src="./images/V100-speedup.jpg" alt="V100加速">

* Tesla P40

<img width="900" height="300" src="./images/p40-perf.jpg" alt="P40性能">
<img width="900" height="300" src="./images/p40-speedup.jpg" alt="P40加速">

* Tesla M40

<img width="900" height="300" src="./images/M40-perf-0302.jpg" alt="M40性能">
<img width="900" height="300" src="./images/M40-speedup-0302.jpg" alt="M40加速">
