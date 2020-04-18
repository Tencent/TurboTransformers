### turbo_transformers: 面向CPU/GPU高效易用的Transformer推理引擎（曾用名fast-transformers）

![logo](./images/logo.jpeg)

**make transformers serving fast by adding a turbo to your inference engine!**

Transformer是近年来NLP领域最重要的模型创新，带来更高的模型精度的同时也引入了更多的计算量，线上Transformer服务的高效部署面临着巨大挑战。面对丰富的Transformer的线上服务场景，微信模式识别中心开源了名为TurboTransformers的Transformer推理加速引擎。Turbo具有如下特性。
1. 优异的CPU/GPU性能表现。面向Intel多核CPU和NVIDIA GPU硬件平台，TurboTransformers可以充发挥硬件的各层级计算能力。在多种CPU和GPU硬件上获得了超过pytorch/tensorflow和目前主流优化引擎（如onnxruntime-mkldnn/onnxruntime-gpu, torch JIT, NVIDIA faster transformers）的性能表现。详细benchmark结果见下文。
2. 为NLP推理任务特点量身定制。和CV任务不同，NLP推理任务输入尺寸多个维度会存在变化。传统做法是补零或者截断成固定长度，这样引入了额外补零计算开销。另外有些框架如onnxruntime、tensorRT、torchlib需要预先对计算图根据输入尺寸进行预处理，这对尺寸变化的NLP任务并不适用。TurboTransformers可以支持变长输入序列处理，且不需要预处理过程。
3. 更简单的使用方式。TurboTransformers支持python和C++接口进行调用。它可以作为pytorch的加速插件，在Transformer任务上，通过加入几行python代码获得的端对端加速效果。

TurboTransformers的已经应用腾讯内部于多个线上BERT服务服务场景。比如，微信的FAQ的服务获得1.88x加速，公有云情感分析服务获得2.11x加速，QQ推荐服务获得13.6x加速。

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
#### 本机构建（编译ONNX-runtime时间会很长）
git clone https://git.code.oa.com/PRC_alg/fast_transformers --recursive
1. 本机构建docker镜像和容器
```
sh tools/build_docker_cpu.sh
# optional: 构建编译环境时需要联网，腾讯内网需要设置代理
export EXTRA_ARGS="--build-arg http_proxy=http://devnet-proxy.oa.com:8080 --build-arg https_proxy=http://devnet-proxy.oa.com:8080"
docker run -it --rm -v your/path/turbo_transformers:/workspace --name=your_container_name REPOSITORY:TAG /bin/bash
cd /workspace
# optional:在编译环境内安装是也需要联网，腾讯内网请设置代理
export http_proxy=http://devnet-proxy.oa.com:8080
export https_proxy=http://devnet-proxy.oa.com:8080
export no_proxy=git.code.oa.com
```

2. 在docker内进行安装
方法1：我想单测和benchmark
```
cd /workspace
# 安装前需要跑一个单测，这里必须下载单测需要的预训练模型，需要git lfs，sudo yum install git-lfs
git lfs install
git lfs pull
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
4. 在docker内安装conda和pip包（optional）

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
docker run --net=host --rm -it -v $PWD:/myspace -v /etc/passwd:/etc/passwd --name=your_container_name REPOSITORY:TAG
# for example: docker run --net=host --rm -it -v $PWD:/myspace -v /etc/passwd:/etc/passwd --name=jiarui_gpu_env ccr.ccs.tencentyun.com/mmspr/turbo_transformers:0.1.1-cuda9.0-ubuntu16.04-gpu-dev
```

2. 在docker内安装pip包并单测
```
cd /myspace
# 下载预训练模型，需要git lfs，sudo yum install git-lfs
git lfs install
git lfs pull

# 可以用TEG机智平台容器的鹅厂小伙伴，可以直接使用我们的镜像
# [ 公司共享镜像 ]g-g-wxg-prc-fast-transformer-cu10:v0.0.1
# 创建于 2020-02-26
# 在TEG的容器里安装时需要能连外网，我把代理地址给大家贴出来
# export no_proxy="tlinux-mirrorlist.tencent-cloud.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,mirrors-tlinux.tencentyun.com,.oa.com,.local"
# export http_proxy=http://star-proxy.oa.com:3128
# export https_proxy=http://star-proxy.oa.com:3128
sh tools/build_and_run_unittests.sh $PWD -DWITH_GPU=ON
```

3. 在docker内运行benchmark (optional), 和pytorch比较
```
cd benchmark
bash gpu_run_benchmark.sh
```


### 使用方法
turbo_transformers提供了简单的C++/python调用接口，我们希望尽最大努力适配多样的上线环境，减轻使用者的开发难度。
#### python接口
提供兼容[huggingface/transformerspytorch](https://github.com/huggingface "pytorch")模型载入方式和python saved模型的载入方式。
tensorflow模型可以转化为pytorch saved模型载入，我们尚未提供示例，读者可自行探索。
我们提供了bert seqence classification的书写方式示例。
参考[./example/python](https://git.code.oa.com/PRC_alg/fast_transformers/tree/develop/example/python "python")的例子。
在工蜂上我们还内部开源了一套可以使用turbo的python severing框架[bertserving](https://git.code.oa.com/PRC_alg/bert-serving/tree/develop "bertserving")供使用者参考，它通过asyncio方式异步响应BERT推理的http请求。
#### C++接口
参考[./example/cpp](https://git.code.oa.com/PRC_alg/fast_transformers/tree/develop/example/cpp "C++")的例子。
C++载入npz格式的模型文件，pytorch saved模型和npz转换的脚本在./tools/convert_huggingface_bert_to_npz.py
我们的例子提供了GPU和两种CPU多线程的调用方式。一种是串行响应BERT计算请求，每次BERT计算使用多线程（omp）方式计算，另一种是多线程并行的响应BERT计算请求，每次BERT计算使用单线程方式的方式。
用户使用时候可以通过add_subdirectory方式连接turbo-transformers，这里提供了一个例子[cmake-usage]("https://git.code.oa.com/jiaruifang/turbo-transformers-cpp" "cmake-usage")。
## 性能

### CPU测试效果
我们在三种CPU硬件平台测试了turbo_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[pytorch-jit](https://pytorch.org/docs/stable/_modules/torch/jit.html "pytorch-jit")和[onnxruntime-mkldnn]( https://github.com/microsoft/onnxruntime "onnxruntime-mkldnn")实现作为对比。性能测试结果为迭代150次的均值。为了避免多次测试时，上次迭代的数据在cache中缓存的现象，每次测试采用随机数据，并在计算后刷新的cache数据。

* Intel Xeon 61xx


在61xx上，四种Transformer实现性能对比结果如下面两张图所示。可以观察到在线程数为1时，四种实现的差别并不大。随着线程数增多，turbo_transformers的性能优势逐步增大，当线程为8时加速效果最为明显。另外，随着seq_length长度增长，turbo_transformers的加速效果减弱，原因是此时GEMM运算时间占比增大，核心融合带来增益减少。

<img width="900" height="300" src="./images/61xx_perf_thd48_0415.jpg" alt="61xx性能">
<img width="900" height="300" src="./images/61xx_speedup_thd48_0415.jpg" alt="61xx加速">

* Intel Xeon 6133

相比61xx型号，Intel Xeon 6133向量化长度更长为512 bit，并且它拥有一个30 MB核间共享L3 cache。如下两张图展示了6133的性能表现。多线程的大部分case，turbo_transformers结果优于其他实现。比较特殊的case是序列长度为10和20的情况。造成这种现象是由于MKL AVX512 GEMM例程的缘故，在Intel 6133 CPU上，我们发现随着seq_length增加，GEMM运算的延迟会出现一个跳变的现象。

<img width="900" height="300" src="./images/6133_perf_thd48_0415.jpg" alt="6133性能">
<img width="900" height="300" src="./images/6133_speedup_thd48_0415.jpg" alt="6133加速">

### GPU测试效果
我们在三种GPU硬件平台测试了turbo_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[NVIDIA Faster Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer "FasterTransformer")，[onnxruntime-gpu](https://github.com/microsoft/onnxruntime "onnxrt-gpu")实现作为对比。性能测试结果为迭代150次的均值。

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

## 技术文档
2020.03.16之前我们的项目曾以fast-transformers发布。
[turbo-transformers (1): CPU Serving is All You Need](http://km.oa.com/group/24938/articles/show/405322?kmref=author_post "turbo-transformers-cpu")
[turbo-transformers (2): GPU Serving Can Also Be You Need](http://km.oa.com/group/18832/articles/show/413605?kmref=author_post "turbo-transformers-gpu")

## 加入用户群
请联系josephyu, jiaruifang, florianzhao加入我们的用户使用群，我们将竭诚为你服务。
