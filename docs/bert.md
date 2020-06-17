We show BERT inference performance here.

### CPU
We tested the performance of TurboTransformers on three CPU hardware platforms.
We choose [pytorch](https://github.com/huggingface "pytorch"), [pytorch-jit](https://pytorch.org/docs/stable/_modules/torch/jit.html "pytorch-jit" ) and [onnxruntime-mkldnn](https://github.com/microsoft/onnxruntime "onnxruntime-mkldnn") and TensorRT implementation as a comparison. The performance test result is the average of 150 iterations. In order to avoid the phenomenon that the data of the last iteration is cached in the cache during multiple tests, each test uses random data and refreshes the cache data after calculation.
* Intel Xeon 61xx

<img width="900" height="300" src="../images/61xx_perf_thd48_0415.jpg" alt="61xx性能">
<img width="900" height="300" src="../images/61xx_speedup_thd48_0415.jpg" alt="61xx加速">

* Intel Xeon 6133
Compared to the 61xx model, Intel Xeon 6133 has a longer vectorized length of 512 bits, and it has a 30 MB shared L3 cache between cores.

<img width="900" height="300" src="../images/6133_perf_thd48_0415.jpg" alt="6133性能">
<img width="900" height="300" src="../images/6133_speedup_thd48_0415.jpg" alt="6133加速">

### GPU
We tested the performance of turbo_transformers on four GPU hardware platforms.
We choose [pytorch](https://github.com/huggingface "pytorch"), [NVIDIA Faster Transformers](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer "FasterTransformer"), [onnxruntime-gpu](https://github.com/microsoft/onnxruntime "onnxrt-gpu") and [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/6.0/demo/BERT) implementation as a comparison. The performance test result is the average of 150 iterations.

* RTX 2060
<img width="900" height="300" src="../images/2060-perf.jpg" alt="2060性能">
<img width="900" height="300" src="../images/2060-speedup.jpg" alt="2060加速">

* Tesla V100

<img width="900" height="300" src="../images/v100-perf.jpg" alt="V100性能">
<img width="900" height="300" src="../images/V100-speedup.jpg" alt="V100加速">

* Tesla P40

<img width="900" height="300" src="../images/p40-perf.jpg" alt="P40性能">
<img width="900" height="300" src="../images/p40-speedup.jpg" alt="P40加速">

* Tesla M40

<img width="900" height="300" src="../images/M40-perf-0302.jpg" alt="M40性能">
<img width="900" height="300" src="../images/M40-speedup-0302.jpg" alt="M40加速">
