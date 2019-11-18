#include "fast_transformers/layers/bert_attention.h"
#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
#include "fast_transformers/layers/kernels/softmax.h"
#include "fast_transformers/layers/kernels/transpose.h"
#include "loguru.hpp"

namespace fast_transformers {
namespace layers {

namespace details {

static void matmul(bool TransA, bool TransB, int m, int n, int k, float alpha,
                   const float* A, int lda, int64_t strideA, const float* B,
                   int ldb, int64_t strideB, const float beta, float* C,
                   int ldc, int64_t strideC, int batchCount) {
  const float* A_Array[batchCount];
  const float* B_Array[batchCount];
  float* C_Array[batchCount];
  for (int i = 0; i < batchCount; ++i) {
    A_Array[i] = A + strideA * i;
    B_Array[i] = B + strideB * i;
    C_Array[i] = C + strideC * i;
  }
  CBLAS_TRANSPOSE transA = TransA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = TransB ? CblasTrans : CblasNoTrans;
  core::Blas().sgemm_batch_(CblasColMajor, &transA, &transB, &m, &n, &k, &alpha,
                            A_Array, &lda, B_Array, &ldb, &beta, C_Array, &ldc,
                            1, &batchCount);
}
}  // namespace details

core::Tensor BertAttention::operator()(const core::Tensor& input_tensor,
                                       const core::Tensor& attention_mask,
                                       const core::Tensor& head_mask) const {
  FT_ENFORCE_EQ(input_tensor.n_dim(), 3,
                "The input ids should be a matrix with shape [BatchSize, "
                "SeqLen, HiddenSize].");
  auto batch_size = input_tensor.shape(0);
  auto seq_length = input_tensor.shape(1);
  auto hidden_size = input_tensor.shape(2);
  auto size_per_head = hidden_size / num_attention_heads_;
  LOG_S(3) << "batch_size: " << batch_size
           << ", num_head: " << num_attention_heads_
           << ", seq_length: " << seq_length << ", hidden_size: " << hidden_size
           << ", size_per_head: " << size_per_head;

  // numel of Q/K/V
  auto buf_size = batch_size * seq_length * hidden_size;
  // numel of Q*K, here from and to has the same lenght
  auto attention_scores_size =
      batch_size * num_attention_heads_ * seq_length * seq_length;

  static core::AlignedScratchpad<float> buf;
  float* buffer = buf.mutable_data(buf_size * 8 + attention_scores_size);

  float* query_buf = buffer;
  float* key_buf = buffer + buf_size;
  float* value_buf = buffer + 2 * buf_size;

  float* q_buf = buffer + 3 * buf_size;
  float* k_buf = buffer + 4 * buf_size;
  float* v_buf = buffer + 5 * buf_size;

  float* self_attr_out = buffer + 6 * buf_size;
  float* context_layer = buffer + 7 * buf_size;
  float* attention_scores = buffer + 8 * buf_size;

  // Let's do forward
  int64_t m = batch_size * seq_length;
  int64_t k = num_attention_heads_ * size_per_head;
  int64_t n = k;
  const float alpha = 1.0f, beta = 0.0f;

  // TODO assert from_tensor is equal to to_tensor.
  // TODO delete the wrapper after we check the results, and rewrite it with
  // Blas().
  core::Tensor output_tensor(
      core::NewDLPackTensorT<float>({batch_size, seq_length, hidden_size}));
  const float* query_weight_ptr = query_weight_.data<float>();
  const float* key_weight_ptr = key_weight_.data<float>();
  const float* value_weight_ptr = value_weight_.data<float>();
  const float* dense_weight_ptr = dense_weight_.data<float>();
  const float* from_tensor_ptr = input_tensor.data<float>();
  const float* to_tensor_ptr = from_tensor_ptr;  // self attention
  const float* query_bias_ptr = query_bias_.data<float>();
  const float* value_bias_ptr = value_bias_.data<float>();
  const float* key_bias_ptr = key_bias_.data<float>();
  const float* dense_bias_ptr = dense_bias_.data<float>();
  float* output_tensor_ptr = output_tensor.mutableData<float>();

  core::Blas().sgemm_(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                      from_tensor_ptr, k, query_weight_ptr, k, beta, query_buf,
                      n);
  core::Blas().sgemm_(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                      to_tensor_ptr, k, key_weight_ptr, k, beta, key_buf, n);

  core::Blas().sgemm_(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                      to_tensor_ptr, k, value_weight_ptr, k, beta, value_buf,
                      n);

  const std::vector<int64_t> QKV_shape{batch_size, seq_length,
                                       num_attention_heads_, size_per_head};
  kernels::AdBiasTransposeForScore(q_buf, query_buf, query_bias_ptr, QKV_shape);
  kernels::AdBiasTransposeForScore(k_buf, key_buf, key_bias_ptr, QKV_shape);
  kernels::AdBiasTransposeForScore(v_buf, value_buf, value_bias_ptr, QKV_shape);

  // attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
  details::matmul(true, false, seq_length, seq_length, size_per_head, alpha,
                  k_buf, size_per_head, seq_length * size_per_head, q_buf,
                  size_per_head, seq_length * size_per_head, beta,
                  attention_scores, seq_length, seq_length * seq_length,
                  batch_size * num_attention_heads_);

  // attention_scores = attention_scores / math.sqrt(self.attention_head_size)
  // attention_scores = attention_scores + attention_mask
  // attention_probs = nn.Softmax(dim=-1)(attention_score)
  const float scaler = 1 / sqrtf(size_per_head * 1.0f);

  kernels::SoftmaxMask(attention_scores, attention_mask.data<float>(),
                       batch_size, num_attention_heads_, seq_length,
                       scaler);  // preprocess: attention_mask * -10000
  // TODO: attention_probs = self.dropout(attention_probs)

  // context_layer = torch.matmul(attention_probs, value_layer) -> context_layer
  details::matmul(false, false, size_per_head, seq_length, seq_length, alpha,
                  v_buf, size_per_head, seq_length * size_per_head,
                  attention_scores, seq_length, seq_length * seq_length, beta,
                  context_layer, size_per_head, seq_length * size_per_head,
                  batch_size * num_attention_heads_);

  // context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
  // new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
  // context_layer = context_layer.view(*new_context_layer_shape)
  kernels::TransposeForScore(
      self_attr_out, context_layer,
      {batch_size, num_attention_heads_, seq_length, size_per_head});

  // self_outputs = (context_layer, attention_probs) if self.output_attentions
  // else (context_layer,) # self.output_attentions is nullptr hidden_states =
  // self.dense(self_outputs[0]) #context_layer
  core::Blas().sgemm_(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                      self_attr_out, k, dense_weight_ptr, k, beta,
                      output_tensor_ptr, n);

  // attention_output = self.LayerNorm(hidden_states + input_tensor)
  kernels::AddBiasLayerNorm(output_tensor_ptr, from_tensor_ptr, dense_bias_ptr,
                            layer_norm_weight_.data<float>(),  // gemma
                            layer_norm_bias_.data<float>(), m, n);

  // outputs = (attention_output,) + self_outputs[1:]
  return output_tensor;
}

void BertAttention::EnforceShapeAndType() const {
  FT_ENFORCE_EQ(query_weight_.device_type(), kDLCPU, "Only CPU supportted");
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
    query_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
    query_bias_.Print<float>(os);
    os << ">>>>>>>>>>>> layer_norm_weight <<<<<<<<<<<<" << std::endl;
    layer_norm_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> layer_norm_bias <<<<<<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace fast_transformers
