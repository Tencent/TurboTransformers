## How to profile you code
1. Before compiling code, set option WITH_PROFILER ON in CMakeLists.txt

```
option(WITH_PROFILER  "Compile with profiler"   ON)
```
2. Add profiling context in your code, for example

```
with turbo_transformers.pref_guard("info") as perf:
      dec_out, dec_attn = self.turbo_decoder(
              decoder_in, memory_bank, memory_lengths=memory_lengths, step=step
          )
```

3. The profiling results will be shown on your screen, like this
```
info Time line:
context/values/AddBiasTransposeForScore/reshape , 0.023328, 0.00835687 %
context/keys/AddBiasTransposeForScore/Reshape , 0.030464, 0.0109132 %
context/gemm2/k_out1/Reshape , 0.040384, 0.0144669 %
context/keys/AddBiasTransposeForScore , 0.04688, 0.016794 %
context/gemm1/v_out1/Reshape , 0.049568, 0.0177569 %
context/values/AddBiasTransposeForScore , 0.050304, 0.0180206 %
context/gemm2 , 0.300832, 0.107768 %
context/gemm1 , 0.322112, 0.115391 %
context/AddBiasTransposeForScore/q_out2/Reshape , 0.515776, 0.184768 %
context/AddBiasTransposeForScore/q_out1/Reshape , 0.616032, 0.220683 %
context/gemm0/q_out2/Reshape , 0.773504, 0.277095 %
self/self_value/Reshape , 0.794784, 0.284718 %
self/layernorm/Reshape , 0.801984, 0.287297 %
FFN/Reshape , 0.851904, 0.30518 %
self/self_key/Reshape , 0.986688, 0.353464 %
FFN/AddBiasAct , 1.28634, 0.460808 %
context/gemm0/q_out1/Reshape , 1.35478, 0.485329 %
self/qkv_out1/Reshape , 1.5048, 0.539069 %
context/AddBiasTransposeForScore , 1.5057, 0.53939 %
self/SplitAddBiasTransposeForScore , 1.56646, 0.561159 %
FFN/LayerNorm , 1.64701, 0.590013 %
gemm5/Reshape , 1.65885, 0.594255 %
context/gemm0/prelayernorm , 1.66512, 0.596501 %
LayerNorm , 1.6856, 0.603838 %
self/self_value/Copy , 1.68688, 0.604297 %
batch_gemm4/Reshape , 1.69667, 0.607804 %
Concat/Reshape , 1.796, 0.643387 %
ApplyMaskAndSoftmax/Reshape , 1.80499, 0.646608 %
batch_gemm3/Reshape , 2.03645, 0.729523 %
Reshape , 2.1289, 0.762641 %
self/layernorm/Copy , 2.53923, 0.909637 %
self/qkv_out2/Reshape , 2.65715, 0.95188 %
Concat , 2.76022, 0.988804 %
context/gemm0/prelayernorm/Copy , 2.83021, 1.01387 %
batch_gemm4 , 3.00442, 1.07628 %
self/self_key/Copy , 3.07203, 1.1005 %
batch_gemm3 , 3.34592, 1.19862 %
ApplyMaskAndSoftmax , 3.67014, 1.31477 %
TransposeForScore , 3.76816, 1.34988 %
FFN/AddInputBias , 3.97325, 1.42335 %
self/keys/Concat , 4.66528, 1.67126 %
self/values/Concat , 5.08947, 1.82322 %
context/gemm0 , 5.76464, 2.06509 %
self/gemm012_fused , 7.82285, 2.8024 %
gemm5 , 11.295, 4.04626 %
FFN/gemm0 , 12.2295, 4.381 %
FFN/gemm1 , 17.4551, 6.25299 %
MultiHeadedAttention_context , 60.8736, 21.8069 %
MultiHeadedAttention_self , 91.1025, 32.6359 %
```
