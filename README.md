### fast_transformers: 面向CPU/GPU高效易用的Transformer推理引擎

Transformer是近两年来NLP领域最重要的模型创新，在带来更高的模型精度的同时也引入了更多的计算量，高效部署Transformer线上服务面临着巨大挑战。面对丰富的Transformer的线上服务场景，微信模式识别中心开源了名为fast_transformers的面向Intel多核CPU和NVIDIA GPU硬件平台的Transformer实现。fast_transformers充发挥硬件的各层级计算能力，并支持变长输入序列处理，避免了补零的额外计算。fast_transformer在多种CPU和GPU硬件上获得了超过pytorch/tensorflow和目前主流优化引擎（如onnxruntime-mkldnn/onnxruntime-gpu, torch JIT, NVIDIA faster transformers）的性能表现，详细benchmark结果见下文，它对常用的短序列的处理速度提升更为显著。fast_transformers CPU版本已经应用于多个线上服务服务场景，WXG的FAQ的BERT服务获得1.88x加速，CSIG的在公有云情感分析服务两层BERT encoder获得2.11x加速。

### CPU版本安装
git clone https://git.code.oa.com/PRC_alg/fast_transformers --recursive
1. 本机构建docker镜像和容器
本机构建（编译ONNX-runtime时间会很长）
```
sh tools/build_docker_cpu.sh
# optional: 构建编译环境时需要联网，腾讯内网需要设置代理
export EXTRA_ARGS="--build-arg http_proxy=http://devnet-proxy.oa.com:8080 --build-arg https_proxy=http://devnet-proxy.oa.com:8080"
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
git clone https://git.code.oa.com/PRC_alg/fast_transformers --recursive
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
# 两种方式载入模型，这里直接从pytorch模型载入
ft_model = fast_transformers.BertModel.from_torch(model)
# 从文件载入
# model = fast_transformers.BertModel.from_pretrained("bert-base-chinese", test_device)
res = ft_model(input_ids)
print(res)
# tensor([[-1.4292,  1.0934, -0.3270,  ...,  0.7212, -0.3893, -0.1172],
#         [-0.8878,  0.6571, -0.6331,  ...,  0.2759, -0.4496, -0.2375]])
```
更多使用接口可以参考 ./benchmark/benchmark.py文件

2. GPU
```python
import os
import numpy
import torch
import transformers
import fast_transformers

torch.set_grad_enabled(False)
test_device = torch.device('cuda:0')
# load model from file, adapted to offline enviroments
model_id = os.path.join(os.path.dirname(__file__),
                         '../fast_transformers/python/tests/test-model')
# model_id = "bert-base-chinese"
model_torch = transformers.BertModel.from_pretrained(model_id)
model_torch.eval()
model_torch.to(test_device)
# the following two ways are the same
# 1. load model from checkpoint in file
# model_ft = fast_transformers.BertModel.from_pretrained(model_id, test_device)
# 2. load model from pytorch model
model_ft = fast_transformers.BertModel.from_torch(model_torch, test_device)
cfg = model_torch.config  # type: transformers.BertConfig

batch_size, seq_len = 10, 40
input_ids = torch.randint(low=0,
                          high=cfg.vocab_size - 1,
                          size=(batch_size, seq_len),
                          dtype=torch.long,
                          device=test_device)

torch_result = model_torch(input_ids)
torch_result = (torch_result[0][:, 0]).cpu().numpy()
print(torch_result)
# [[-1.0547106   0.6978769   0.01930561 ...  0.14942119  0.12424274
#    0.09840814]
#  [-0.38007614  0.8337314  -0.26551855 ...  0.37165833 -1.1472359
#   -0.02053148]
#  [-1.0879238   0.33059713 -1.5077729  ...  1.1362088  -1.1507283
#    0.72761345]
#  ...
#  [-0.13508567  0.5811261  -0.7433949  ...  0.787879    0.19001244
#    0.2780586 ]
#  [-0.2851665   1.0065655  -0.15112075 ... -0.39093181  0.07916196
#   -0.2520734 ]
#  [-0.56147367  0.6245851  -0.97631836 ...  1.300955    0.2231751
#   -0.35811383]]
  
ft_result = model_ft(input_ids)
ft_result = ft_result.cpu().numpy()
print(ft_result)
# [[-1.0559868   0.70307076  0.01852467 ...  0.15004729  0.12565975
#    0.0958093 ]
#  [-0.37808716  0.8361676  -0.26031977 ...  0.36978582 -1.150511
#   -0.02066079]
#  [-1.0828258   0.32773003 -1.5136379  ...  1.1391504  -1.1485292
#    0.7275925 ]
#  ...
#  [-0.13846944  0.5822965  -0.7396279  ...  0.78931236  0.19155282
#    0.27686128]
#  [-0.28799683  1.007286   -0.15279569 ... -0.38764566  0.07981472
#   -0.2519405 ]
#  [-0.5681539   0.62843543 -0.982041   ...  1.2941586   0.22365712
#   -0.3616636 ]]

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
我们选择[pytorch](https://github.com/huggingface "pytorch")，[NVIDIA Faster Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer "FasterTransformer")，[onnxruntime-gpu](https://github.com/microsoft/onnxruntime "onnxrt-gpu")实现作为对比。性能测试结果为迭代150次的均值。


* Tesla V100

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202002/1582869887_97_w2182_h1030.png" alt="V100性能">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202002/1582878605_32_w2180_h1022.png" alt="V100加速">

* Tesla P40

<img width="600" height="180" src="http://km.oa.com/files/photos/captures/202003/1583486895_54_w3084_h768.png" alt="P40性能">
<img width="600" height="180" src="http://km.oa.com/files/photos/captures/202003/1583486872_73_w3094_h888.png" alt="P40加速">


* Tesla M40

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579437911_33_w2828_h1322.png" alt="M40性能短序列">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579438488_16_w2182_h1008.png" alt="M40加速短序列">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579509883_75_w2916_h1348.png" alt="M40性能长序列">

* Tesla vs CPU

<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579506844_27_w2786_h1302.png" alt="M40vsCPU">
<img width="600" height="300" src="http://km.oa.com/files/photos/captures/202001/1579507047_29_w2844_h1338.png" alt="M40vsCPU">

## 技术文档
[fast-transformers (1): CPU Serving is All You Need](http://km.oa.com/group/24938/articles/show/405322?kmref=author_post "fast-transformers-cpu")
[fast-transformers (2): GPU Serving Can Also Be You Need](http://km.oa.com/group/18832/articles/show/413605?kmref=author_post "fast-transformers-gpu")