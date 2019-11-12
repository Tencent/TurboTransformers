#pragma once
#include <stdint.h>
#include <cmath>
#include <numeric>
#include <vector>

namespace fast_transformers {
namespace layers {
namespace kernels {

/***
 * Transpose from (Batch, Seq_len, Head, Size_per_head) ->
 * (Batch, Head, Seq_len, Size_per_head)
 * def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor
 * **/

extern void TransposeForScore(float* output, const float* input,
                              const std::vector<int64_t>& shape);

extern void AdBiasTransposeForScore(float* output, const float* input,
                                    const float* bias,
                                    const std::vector<int64_t>& shape);

// input (batch_size, seq_length, 3, head_num, *size_per_head)
//(3, batch_size, head_num, seq_length, *size_per_head)
// bias: (3, head_num, size_per_head)

extern void SplitAddbiasTransposeForScore(float* output, const float* input,
                                          const float* bias,
                                          const std::vector<int64_t>& shape);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
