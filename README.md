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
