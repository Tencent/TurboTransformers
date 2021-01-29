// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "turbo_transformers/core/allocator/bert_config.h"

#include <algorithm>
#include <limits>
#include <memory>

namespace turbo_transformers {
namespace core {
namespace allocator {
namespace bert_config {

/*
 * build a tensor usage recorders for bert. A trick to reduce the overhead of
 * later memory arrangement is that we make the tensor arrangement of all layers
 * to be the same.
 * @Inputs : parameters to define a BERT.
 * @Outputs : tensor usage records (TURs) (name, start_op, end_op, size)
 */

#define ADDITEM(a, b, c, d)                                  \
  TensorUsageRecord.emplace_back(                            \
      std::make_shared<TensorRecordItem>(a, (b), (c), (d))); \
  activation_set.insert(a);

/*
 * TODO(jiaruifang) The tensor usage record should be generated automatically
 * A practical method is to run a DNN inference and trace the allocate and
 * free time of each tensor.
 */
template <typename T>
void GetBertTensorUsageRecord(
    std::vector<TensorRecordItemPtr>& TensorUsageRecord,
    std::set<std::string>& activation_set, int64_t batch_size, int64_t seq_len,
    int64_t num_head, int64_t hidden_size, int64_t num_layer) {
  TensorUsageRecord.clear();
  activation_set.clear();

  auto item_bytes = sizeof(T);
  auto id_bytes = sizeof(int64_t);
  auto from_seq_len = seq_len;
  auto to_seq_len = seq_len;

  auto Pooler_size = batch_size * hidden_size * item_bytes;
  auto seq_pool_size = hidden_size * item_bytes;
  auto Q_size = batch_size * from_seq_len * hidden_size * item_bytes;
  auto K_size = batch_size * to_seq_len * hidden_size * item_bytes;
  auto V_size = K_size;
  auto attn_score_size =
      batch_size * num_head * from_seq_len * to_seq_len * item_bytes;
  auto aligned_id_seq_size = from_seq_len * batch_size * id_bytes;

  auto extendedattnmask_size = batch_size * from_seq_len * item_bytes;
  ADDITEM("PrepareBertMasks/possitionids/Reshape", 0, 1, aligned_id_seq_size);
  ADDITEM("PrepareBertMasks/seqids/Reshape", 0, 1, aligned_id_seq_size);
  ADDITEM("PrepareBertMasks/attmask/Reshape", 0, 0, aligned_id_seq_size);
  ADDITEM("PrepareBertMasks/extendedattnmask/Reshape", 0, 11,
          extendedattnmask_size);
  // NOTE: bert embedding overlap with output
  ADDITEM("BERTEmbedding/Reshape", 1, 11, Q_size);

  int64_t start_idx = 2;
  ADDITEM("self/qkv_out1/Reshape", start_idx + 0, start_idx + 1,
          K_size + Q_size + V_size);
  ADDITEM("self/q/Reshape", start_idx + 1, start_idx + 2, Q_size);
  ADDITEM("self/k/Reshape", start_idx + 1, start_idx + 2, K_size);
  ADDITEM("self/v/Reshape", start_idx + 1, start_idx + 3, V_size);
  ADDITEM("batch_gemm3/Reshape", start_idx + 2, start_idx + 3, attn_score_size);
  ADDITEM("ApplyMaskAndSoftmax/Reshape", start_idx + 3, start_idx + 4, Q_size);
  ADDITEM("batch_gemm4/Reshape", start_idx + 4, start_idx + 5, attn_score_size);
  ADDITEM("gemm5/Reshape", start_idx + 5, start_idx + 8, Q_size);
  ADDITEM("BertIntermediate/Reshape", start_idx + 7, start_idx + 8, Q_size * 4);
  // BertOutput/Reshape is the same as BERTEmbedding/Reshape
  //  ADDITEM("BertOutput/Reshape", start_idx + 0, start_idx + 9, Q_size);
  ADDITEM("BertPooler", start_idx + 9, start_idx + 10, Pooler_size);
  ADDITEM("SeqPool", start_idx + 10, start_idx + 11, seq_pool_size);

  // sort descend order
  std::sort(
      TensorUsageRecord.begin(), TensorUsageRecord.end(),
      [](const auto& a, const auto& b) -> bool { return a->size_ > b->size_; });
}
#undef ADDITEM

template void GetBertTensorUsageRecord<float>(
    std::vector<TensorRecordItemPtr>& TensorUsageRecord,
    std::set<std::string>& activation_set, int64_t batch_size, int64_t seq_len,
    int64_t num_head, int64_t hidden_size, int64_t num_layer);

}  // namespace bert_config
}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
