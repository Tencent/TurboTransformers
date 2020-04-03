### turbo_transformers: 面向CPU/GPU高效易用的Transformer推理引擎（曾用名fast-transformers）

***make transformers serving fast easily by adding turbo to your inference engine!***

Transformer是近两年来NLP领域最重要的模型创新，在带来更高的模型精度的同时也引入了更多的计算量，高效部署Transformer线上服务面临着巨大挑战。面对丰富的Transformer的线上服务场景，微信模式识别中心开源了名为turbo_transformers的面向Intel多核CPU和NVIDIA GPU硬件平台的Transformer实现。turbo_transformers充发挥硬件的各层级计算能力，并支持变长输入序列处理，避免了补零的额外计算。turbo_transformers在多种CPU和GPU硬件上获得了超过pytorch/tensorflow和目前主流优化引擎（如onnxruntime-mkldnn/onnxruntime-gpu, torch JIT, NVIDIA faster transformers）的性能表现，详细benchmark结果见下文，它对常用的短序列的处理速度提升更为显著。turbo_transformers CPU版本已经应用于多个线上服务服务场景，WXG的FAQ的BERT服务获得1.88x加速，CSIG的在公有云情感分析服务两层BERT encoder获得2.11x加速。

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
git clone https://git.code.oa.com/PRC_alg/fast_transformers --recursive
1. 本机构建docker镜像和容器
本机构建（编译ONNX-runtime时间会很长）
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

2. 在docker内安装conda和pip包

```
sh tool/build_conda_package.sh
# conda包会在 /workspace/dist/*.tar.bz2中
# 在本容器外其他环境使用turbo_transformers时只需要python -m pip install your_root_path/dist/*.tar.bz2
```

3. 在docker内进行单测 (optional)

```
cd /workspace
# 下载预训练模型，需要git lfs，sudo yum install git-lfs
git lfs install
git lfs pull
sh tools/build_run_unittests.sh $PWD -DWITH_GPU=OFF
```
4. 在docker内运行benchmark (optional), 和pytorch, torch-JIT, onnxruntime比较

```
cd benchmark
bash run_benchmark.sh
```

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
提供兼容huggingface/transformers[pytorch](https://github.com/huggingface "pytorch")模型载入方式和python saved模型的载入方式。
tensorflow模型可以转化为pytorch saved模型载入，我们尚未提供示例，读者可自行探索。
我们提供了bert seqence classification的书写方式示例。
参考./example/python [python](https://git.code.oa.com/PRC_alg/fast_transformers/raw/develop/example/pythonce "python")的例子。
在工蜂上我们还内部开源了一套可以使用turbo的python severing框架[bertserving](https://git.code.oa.com/PRC_alg/bert-serving/tree/develop "bertserving")供使用者参考，它通过asyncio方式异步响应BERT推理的http请求。
#### C++接口
参考./example/cpp [C++](https://git.code.oa.com/PRC_alg/fast_transformers/raw/develop/example/cpp "C++")的例子。
C++载入npz格式的模型文件，pytorch saved模型和npz转换的脚本在./tools/convert_huggingface_bert_to_npz.py
我们的例子两种调用方式。一种是串行响应BERT计算请求，每次BERT计算使用多线程（omp）方式计算，另一种是多线程并行的响应BERT计算请求，每次BERT计算使用单线程方式的方式。

## 性能

### CPU测试效果
我们在三种CPU硬件平台测试了turbo_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[pytorch-jit](https://pytorch.org/docs/stable/_modules/torch/jit.html "pytorch-jit")和[onnxruntime-mkldnn]( https://github.com/microsoft/onnxruntime "onnxruntime-mkldnn")实现作为对比。性能测试结果为迭代150次的均值。为了避免多次测试时，上次迭代的数据在cache中缓存的现象，每次测试采用随机数据，并在计算后刷新的cache数据。


* Intel Xeon 61xx


在61xx上，四种Transformer实现性能对比结果如下面两张图所示。可以观察到在线程数为1时，四种实现的差别并不大。随着线程数增多，turbo_transformers的性能优势逐步增大，当线程为8时加速效果最为明显。另外，随着seq_length长度增长，turbo_transformers的加速效果减弱，原因是此时GEMM运算时间占比增大，核心融合带来增益减少。

<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584350217_86_w3088_h1026.png">
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584350234_3_w3104_h1026.png" alt="61xx性能2">

* Intel Xeon 6133

相比61xx型号，Intel Xeon 6133向量化长度更长为512 bit，并且它拥有一个30 MB核间共享L3 cache。如下两张图展示了6133的性能表现。多线程的大部分case，turbo_transformers结果优于其他实现。比较特殊的case是序列长度为10和20的情况。造成这种现象是由于MKL AVX512 GEMM例程的缘故，在Intel 6133 CPU上，我们发现随着seq_length增加，GEMM运算的延迟会出现一个跳变的现象。

<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584350279_35_w3092_h1028.png" alt="6133性能1">
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584350292_90_w3104_h1012.png" alt="6133性能2">


* intel i9-9800 CPU

如下两张图展示了intel i9上的性能表现。再线程数大于1时，turbo_transformers的性能优于其他实现。
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584350313_80_w3090_h1020.png" alt="6133性能1">
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584350326_14_w3088_h1030.png" alt="6133性能2">

### GPU测试效果
我们在三种GPU硬件平台测试了turbo_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[NVIDIA Faster Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer "FasterTransformer")，[onnxruntime-gpu](https://github.com/microsoft/onnxruntime "onnxrt-gpu")实现作为对比。性能测试结果为迭代150次的均值。


* Tesla V100

<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584351870_55_w3094_h1016.png" alt="V100性能">
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584351683_62_w3086_h1030.png" alt="V100加速">

* Tesla P40

<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584351888_63_w3082_h1016.png" alt="P40性能">
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584351721_73_w3082_h1012.png" alt="P40加速">


* Tesla M40

<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584351914_10_w3096_h1030.png" alt="M40性能短序列">
<img width="900" height="300" src="http://km.oa.com/files/photos/captures/202003/1584351928_90_w3098_h1018.png" alt="M40加速短序列">

## 技术文档
2020.03.16之前我们的项目曾以fast-transformers发布。
[turbo-transformers (1): CPU Serving is All You Need](http://km.oa.com/group/24938/articles/show/405322?kmref=author_post "turbo-transformers-cpu")
[turbo-transformers (2): GPU Serving Can Also Be You Need](http://km.oa.com/group/18832/articles/show/413605?kmref=author_post "turbo-transformers-gpu")

## 加入用户群
请联系josephyu, jiauifang, florianzhao加入我们的用户使用群，我们将竭诚为您服务。
