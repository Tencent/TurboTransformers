### fast_transformers: 基于CPU的快速Transformer推理工具

Transformer是近两年来NLP领域最重要的模型创新，在带来更高的模型精度的同时也引入了更多的计算量，这对高效部署Transformer线上服务带来了巨大挑战。面对丰富的Transformer的线上服务场景，微信模式识别中心开源了名为fast_transformers的基于Intel多核CPU的Transformer实现。fast_transformers充发挥CPU的核间并行和指令级并行能力，并支持变长输入序列处理，避免了补零的额外计算。fast_transformer在多种CPU硬件上获得了超过pytorch和目前主流优化引擎（如onnxruntime-mkldnn 和torch JIT）的性能表现。使用4~8个线程时，fast_transformer在10~128长度的序列处理任务中，相比已开源最优实现取得平均20%以上的加速效果。并且，它对短序列的处理速度提升更为显著。fast_transformers已经应用于模式识别中心的多个线上服务服务场景。

### 安装
1.1 本机构建docker镜像和容器
本机构建（编译ONNX-runtime时间会很长）
```
sh tools/build_docker.sh
docker run -it --rm -v your/path/fast_transformers:/workspace --name=your_container_name REPOSITORY:TAG /bin/bash
cd /workspace
# optional:安装需要联网，腾讯内网请设置代理
export http_proxy=http://devnet-proxy.oa.com:8080
export https_proxy=http://devnet-proxy.oa.com:8080
export no_proxy=git.code.oa.com
```
1.2 从腾讯云的dockerhub上pull已有镜像
A. WXG的腾讯云dockerhub账号
```
docker pull ccr.ccs.tencentyun.com/mmspr/fast_transformer:0.1.1a1-dev
```
B. CSIG的腾讯云dockerhub账号
```
docker pull ccr.ccs.tencentyun.com/mmspr/fast_transformer:0.1.1a1-dev
```
2. 在docker内安装conda和pip包

```
sh conda/build.sh
```

3. 在docker内进行单测 (optional)

```
sh tools/build_run_unittests.sh $PWD
```
4. 在docker内运行benchmark (optional)

```
cd benchmark
bash run_benchmark.sh
```

### 使用示例
Fast_transformers提供了简单的调用接口，提供兼容huggingface/transformers [pytorch](https://github.com/huggingface "pytorch")模型的调用方式。
下面代码片段展示了如何将huggingface预训练BERT模型导入fast_transformer并进行一次BERT encoder的计算。

```python
import torch
import transformers
import contexttimer
import fast_transformers
# 使用4个线程运行fast_transformers
fast_transformers.set_num_threads(4)
# 调用transformers提供的预训练模型
model = transformers.BertModel.from_pretrained(
    "bert-base-chinese")
# 预训练模型的配置
cfg = model.config
#随机生成文本序列
input_ids = torch.randint(low=0,
                            high=cfg.vocab_size - 1,
                            size=(batch_size, seq_len),
                            dtype=torch.long)
# 构建bert-encoder的模型，输出first方式pooling的结果
model = fast_transformers.BertModel.from_torch(model, pooling_type=PoolingType.FIRST)
# 获得encoder结果
res = model(input_ids)
```
更多使用接口可以参考 ./benchmark/benchmark.py文件


## 性能
我们在三种硬件平台测试了fast_transformers的性能表现。
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
