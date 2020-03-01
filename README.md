### fast_transformers: 面向CPU/GPU高效易用的Transformer推理引擎

Transformer是近两年来NLP领域最重要的模型创新，在带来更高的模型精度的同时也引入了更多的计算量，高效部署Transformer线上服务面临着巨大挑战。面对丰富的Transformer的线上服务场景，微信模式识别中心开源了名为fast_transformers的面向Intel多核CPU和NVIDIA GPU硬件平台的Transformer实现。fast_transformers充发挥硬件的各层级计算能力，并支持变长输入序列处理，避免了补零的额外计算。fast_transformer在多种CPU和GPU硬件上获得了超过pytorch/tensorflow和目前主流优化引擎（如onnxruntime-mkldnn 和torch JIT）的性能表现。在CPU上，使用4/8个线程时，fast_transformers在10-128长度的序列处理任务中，相比已开源最优实现取得平均20%以上的加速效果。在GPU V100上，fast_transformers在10-128长度的序列处理任务中，相比已开源最优实现取得平均40%的加速效果。并且，它对短序列的处理速度提升更为显著。fast_transformers已经应用于模式识别中心的多个线上服务服务场景。

### CPU版本安装
1. 本机构建docker镜像和容器
本机构建（编译ONNX-runtime时间会很长）
```
sh tools/build_docker_cpu.sh
# optional: 构建编译环境时需要联网，腾讯内网需要设置代理
export EXTRA_ARGS="--build-arg http_proxy=http://devnet-proxy.oa.com:8080 --build-arg https_proxy=http:/ /devnet-proxy.oa.com:8080"
docker run -it --rm -v your/path/fast_transformers:/workspace --name=your_container_name REPOSITORY:TAG /bin/bash
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
# 在本容器外其他环境使用fast_transformer时只需要python -m pip install your_root_path/dist/*.tar.bz2
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
1. 本机构建docker镜像和容器
```
# 可以在脚本中修改环境变量指定cuda版本和操作系统版本
sh tools/build_docker_gpu.sh $PWD
docker run --net=host --rm -it -v $PWD:/myspace -v /etc/passwd:/etc/passwd --name=your_container_name REPOSITORY:TAG
# for example: docker run --net=host --rm -it -v $PWD:/myspace -v /etc/passwd:/etc/passwd --name=jiarui_gpu_env ccr.ccs.tencentyun.com/mmspr/fast_transformer:0.1.1-cuda9.0-ubuntu16.04-gpu-dev
```

2. 在docker内安装pip包并单测
```
cd /myspace
# 下载预训练模型，需要git lfs，sudo yum install git-lfs
git lfs install
git lfs pull
sh tools/build_and_run_unittests.sh $PWD -DWITH_GPU=ON
```

3. 在docker内运行benchmark (optional), 和pytorch比较
```
cd benchmark
bash gpu_run_benchmark.sh
```


### 使用示例
fast_transformers提供了简单的调用接口，提供兼容huggingface/transformers [pytorch](https://github.com/huggingface "pytorch")模型的调用方式。
下面代码片段展示了如何将huggingface预训练BERT模型导入fast_transformer并进行一次BERT encoder的计算。

1. CPU
```python
import torch
import transformers
import fast_transformers

# 使用4个线程运行fast_transformers
fast_transformers.set_num_threads(4)
# 调用transformers提供的预训练模型
model = transformers.BertModel.from_pretrained(
    "bert-base-chinese")
model.eval()
# 预训练模型的配置
cfg = model.config
batch_size = 2
seq_len = 128
#随机生成文本序列
torch.manual_seed(1)
input_ids = torch.randint(low=0,
                            high=cfg.vocab_size - 1,
                            size=(batch_size, seq_len),
                            dtype=torch.long)

torch.set_grad_enabled(False)
torch_res = model(input_ids) # sequence_output, pooled_output, (hidden_states), (attentions)
print(torch_res[0][:,0,:])  # 获取encoder得到的第一个隐状态
# tensor([[-1.4238,  1.0980, -0.3257,  ...,  0.7149, -0.3883, -0.1134],
#        [-0.8828,  0.6565, -0.6298,  ...,  0.2776, -0.4459, -0.2346]])

# 构建bert-encoder的模型，输出first方式pooling的结果
ft_model = fast_transformers.BertModel.from_torch(model)
res = ft_model(input_ids)
print(res)
# tensor([[-1.4292,  1.0934, -0.3270,  ...,  0.7212, -0.3893, -0.1172],
#         [-0.8878,  0.6571, -0.6331,  ...,  0.2759, -0.4496, -0.2375]])
```
更多使用接口可以参考 ./benchmark/benchmark.py文件

2. GPU
```pytorch
import torch
import transformers
import fast_transformers
test_device = torch.device('cuda:0')

model = transformers.BertModel.from_pretrained(
    model)  # type: transformers.BertModel
model.to(test_device)
cfg = model.config  # type: transformers.BertConfig
input_ids = torch.randint(low=0,
                          high=cfg.vocab_size - 1,
                          size=(batch_size, seq_len),
                          dtype=torch.long,
                          device=test_device)
model = fast_transformers.BertModel.from_torch(model)
model(input_ids)

```


## 性能
### CPU测试效果
我们在三种CPU硬件平台测试了fast_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[pytorch-jit](https://pytorch.org/docs/stable/_modules/torch/jit.html "pytorch-jit")和[onnxruntime-mkldnn]( https://github.com/microsoft/onnxruntime "onnxruntime-mkldnn")实现作为对比。性能测试结果为迭代150次的均值。为了避免多次测试时，上次迭代的数据在cache中缓存的现象，每次测试采用随机数据，并在计算后刷新的cache数据。


* Intel Xeon 61xx


在61xx上，四种Transformer实现性能对比结果如下面两张图所示。可以观察到在线程数为1时，四种实现的差别并不大。随着线程数增多，fast_transformers的性能优势逐步增大，当线程为8时加速效果最为明显。另外，随着seq_length长度增长，fast_transformers的加速效果减弱，原因是此时GEMM运算时间占比增大，核心融合带来增益减少。

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/201912/1575381653_78_w1546_h784.png" alt="61xx性能1">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/201912/1575382049_44_w1676_h845.png" alt="61xx性能2">

* Intel Xeon 6133

相比61xx型号，Intel Xeon 6133向量化长度更长为512 bit，并且它拥有一个30 MB核间共享L3 cache。如下两张图展示了6133的性能表现。多线程的大部分case，fast_transformers结果优于其他实现。比较特殊的case是序列长度为10和20的情况。造成这种现象是由于MKL AVX512 GEMM例程的缘故，在Intel 6133 CPU上，我们发现随着seq_length增加，GEMM运算的延迟会出现一个跳变的现象。

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/201912/1575384757_71_w1751_h886.png" alt="6133性能1">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/201912/1575385675_63_w1602_h804.png" alt="6133性能2">


* intel i9-9800 CPU

如下两张图展示了intel i9上的性能表现。再线程数大于1时，fast_transformer的性能优于其他实现。
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/201912/1575425550_58_w2920_h1474.png" alt="6133性能1">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/201912/1575425573_3_w3042_h1534.png" alt="6133性能2">

### GPU测试效果
我们在三种GPU硬件平台测试了fast_transformers的性能表现。
我们选择[pytorch](https://github.com/huggingface "pytorch")，[NVIDIA Faster Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer "FasterTransformer")实现作为对比。性能测试结果为迭代150次的均值。


* Tesla V100

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202002/1582869887_97_w2182_h1030.png" alt="V100性能">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202002/1582878605_32_w2180_h1022.png" alt="V100加速">

* Tesla P40

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202002/1582890431_98_w2182_h1010.png" alt="P40性能">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202002/1582903489_74_w2176_h1018.png" alt="P40加速">


* Tesla M40

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579437911_33_w2828_h1322.png" alt="M40性能短序列">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579438488_16_w2182_h1008.png" alt=M40加速短序列">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579509883_75_w2916_h1348.png" alt="M40性能长序列">

* Tesla vs CPU

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579506844_27_w2786_h1302.png" alt="M40vsCPU">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579507047_29_w2844_h1338.png" alt="M40vsCPU">