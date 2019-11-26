#include "fast_transformers/layers/bert_attention.h"

#include "fast_transformers/core/aligned_scratchpad.h"
#include "fast_transformers/core/memory.h"
#include "fast_transformers/layers/kernels/layer_norm.h"
#include "fast_transformers/layers/kernels/mat_mul.h"
#include "fast_transformers/layers/kernels/softmax.h"
#include "fast_transformers/layers/kernels/transpose.h"
#include "loguru.hpp"
namespace fast_transformers {
namespace layers {

namespace details {

static void matmul(bool TransA, bool TransB, BlasInt m, BlasInt n, BlasInt k,
                   float alpha, const float* A, BlasInt lda, int64_t strideA,
                   const float* B, BlasInt ldb, int64_t strideB,
                   const float beta, float* C, BlasInt ldc, int64_t strideC,
                   BlasInt batchCount) {
  const float* A_Array[batchCount];
  const float* B_Array[batchCount];
  float* C_Array[batchCount];
  for (int i = 0; i < batchCount; ++i) {
    A_Array[i] = A + strideA * i;
    B_Array[i] = B + strideB * i;
    C_Array[i] = C + strideC * i;
  }
  auto transA = TransA ? CblasTrans : CblasNoTrans;
  auto transB = TransB ? CblasTrans : CblasNoTrans;
  cblas_sgemm_batch(CblasColMajor, &transA, &transB, &m, &n, &k, &alpha,
                    A_Array, &lda, B_Array, &ldb, &beta, C_Array, &ldc, 1,
                    &batchCount);
}

static void BatchMatMul(const core::Tensor& A, bool a_trans,
                        const core::Tensor& B, bool b_trans, float alpha,
                        core::Tensor* C, float beta) {
  auto* A_shape = &A.shape(0);
  auto A_ndim = A.n_dim();
  auto* B_shape = &B.shape(0);
  auto B_ndim = B.n_dim();
  FT_ENFORCE_GT(A_ndim, 2, "A must at least be 3 dims");
  FT_ENFORCE_GT(B_ndim, 2, "B must at least be 3 dims");

  int a_rows = A_shape[A_ndim - 2];
  int a_cols = A_shape[A_ndim - 1];
  int b_rows = B_shape[B_ndim - 2];
  int b_cols = B_shape[B_ndim - 1];

  int a_batch_size =
      std::accumulate(A_shape, A_shape + A_ndim - 2, 1, std::multiplies<>());
  int b_batch_size =
      std::accumulate(B_shape, B_shape + B_ndim - 2, 1, std::multiplies<>());

  FT_ENFORCE_EQ(a_batch_size, b_batch_size, "BatchSize mismatch");

  int M = a_trans ? a_cols : a_rows;
  int N = b_trans ? b_rows : b_cols;
  int K_a = a_trans ? a_rows : a_cols;
  int K_b = b_trans ? b_cols : b_rows;
  FT_ENFORCE_EQ(K_a, K_b, "K mismatch");

  auto* C_shape = &C->shape(0);
  auto C_ndim = C->n_dim();

  int c_rows = C_shape[C_ndim - 2];
  int c_cols = C_shape[C_ndim - 1];
  int c_batch_size =
      std::accumulate(C_shape, C_shape + C_ndim - 2, 1, std::multiplies<>());

  FT_ENFORCE_EQ(c_rows, M, "C shape mismatch");
  FT_ENFORCE_EQ(c_cols, N, "C shape mismatch");
  FT_ENFORCE_EQ(c_batch_size, b_batch_size, "C BatchSize mismatch");

  int offsetA = a_rows * a_cols;
  int offsetB = b_rows * b_cols;
  int offsetC = c_rows * c_cols;

  std::unique_ptr<const float* []> A_array(new const float*[a_batch_size]);
  std::unique_ptr<const float* []> B_array(new const float*[b_batch_size]);
  std::unique_ptr<float* []> C_array(new float*[c_batch_size]);

  auto* a_ptr = A.data<float>();
  auto* b_ptr = B.data<float>();
  auto* c_ptr = C->mutableData<float>();

  for (int i = 0; i < a_batch_size; ++i) {
    A_array[i] = a_ptr + i * offsetA;
    B_array[i] = b_ptr + i * offsetB;
    C_array[i] = c_ptr + i * offsetC;
  }
  auto transA = a_trans ? CblasTrans : CblasNoTrans;
  auto transB = b_trans ? CblasTrans : CblasNoTrans;

  int lda = (transA == CblasNoTrans) ? K_a : M;
  int ldb = (transB == CblasNoTrans) ? N : K_a;
  int ldc = N;

  cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &M, &N, &K_a, &alpha,
                    A_array.get(), &lda, B_array.get(), &ldb, &beta,
                    C_array.get(), &ldc, 1, &a_batch_size);
}

}  // namespace details

void BertAttention::operator()(const core::Tensor& input_tensor,
                               const core::Tensor& attention_mask,
                               core::Tensor* output) const {
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

  // allocate memory for temporary buffers
  static core::AlignedScratchpad<float> buf;
  float* buffer = buf.mutable_data(buf_size * 9 + attention_scores_size);

  static core::Tensor query_buffer(nullptr);
  query_buffer.Reshape<float>({3, batch_size, seq_length, hidden_size});
  static core::Tensor qkv_tensor(nullptr);
  qkv_tensor.Reshape<float>(
      {3, batch_size, num_attention_heads_, seq_length, size_per_head});

  static core::Tensor attention_scores_tensor(nullptr);
  attention_scores_tensor.Reshape<float>(
      {batch_size, num_attention_heads_, seq_length, seq_length});

  float* self_attr_out = buffer + 6 * buf_size;
  float* context_layer = buffer + 7 * buf_size;

  // Let's do forward
  int64_t m = batch_size * seq_length;
  int64_t k = num_attention_heads_ * size_per_head;
  int64_t n = k;
  static float alpha = 1., beta = 0.;

  // TODO assert from_tensor is equal to to_tensor.
  // TODO delete the wrapper after we check the results, and rewrite it with
  // Blas().

  FT_ENFORCE(output, "The output tensor should not be nullptr.");
  output->Reshape<float>({batch_size, seq_length, hidden_size});

  auto* dense_weight_ptr = dense_weight_.data<float>();

  auto* output_tensor_ptr = output->mutableData<float>();

  kernels::Matmul(input_tensor, false, qkv_weight_, true, 1.0, &query_buffer,
                  0.0);

  kernels::SplitAddBiasTransposeForScore(&qkv_tensor, query_buffer, qkv_bias_);

  auto q_tensor = qkv_tensor[0];
  auto k_tensor = qkv_tensor[1];
  auto v_tensor = qkv_tensor[2];

  // attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
  details::BatchMatMul(q_tensor, false, k_tensor, true, 1.0,
                       &attention_scores_tensor, 0.0);

  const float scaler = 1 / sqrtf(size_per_head * 1.0f);

  kernels::SoftmaxMask<float>(attention_scores_tensor.mutableData<float>(),
                              attention_mask.data<float>(), batch_size,
                              num_attention_heads_, seq_length,
                              scaler);  // preprocess: attention_mask * -10000
  // attention_probs = self.dropout(attention_probs)
  // TODO

  // context_layer = torch.matmul(attention_probs, value_layer) -> context_layer
  details::matmul(
      false, false, size_per_head, seq_length, seq_length, alpha,
      v_tensor.data<float>(), size_per_head, seq_length * size_per_head,
      attention_scores_tensor.data<float>(), seq_length,
      seq_length * seq_length, beta, context_layer, size_per_head,
      seq_length * size_per_head, batch_size * num_attention_heads_);

  // context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
  // new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
  // context_layer = context_layer.view(*new_context_layer_shape)
  kernels::TransposeForScore<float>(
      self_attr_out, context_layer,
      {batch_size, num_attention_heads_, seq_length, size_per_head});

  // self_outputs = (context_layer, attention_probs) if self.output_attentions
  // else (context_layer,) # self.output_attentions is nullptr hidden_states =
  // self.dense(self_outputs[0]) #context_layer
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
              self_attr_out, k, dense_weight_ptr, k, beta, output_tensor_ptr,
              n);

  // attention_output = self.LayerNorm(hidden_states + input_tensor)
  kernels::AddBiasLayerNorm<float>(input_tensor, dense_bias_,
                                   layer_norm_weight_,  // gemma
                                   layer_norm_bias_, output);
  // outputs = (attention_output,) + self_outputs[1:]
}

void BertAttention::EnforceShapeAndType() const {
  FT_ENFORCE_EQ(layer_norm_weight_.device_type(), kDLCPU,
                "Only CPU supportted");
  if (loguru::current_verbosity_cutoff() > 3) {
    std::ostringstream os;
    layer_norm_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> layer_norm_bias <<<<<<<<<<<<" << std::endl;
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace fast_transformers
