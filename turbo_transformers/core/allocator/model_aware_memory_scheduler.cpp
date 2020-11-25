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

#include "turbo_transformers/core/allocator/model_aware_memory_scheduler.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

namespace turbo_transformers {
namespace core {
namespace allocator {

/**
 * Fit the tensor record into the chunk
 * @param t : the tensor record to be fitted in the chunk
 * @param chunk : a chunk of memory, which has be assigned a set of tensor
 * records It will be updated if reture is true.
 * @return : whether or not the chunk has fit the tensor record.
 */
static bool TryFitChunk(
    const std::shared_ptr<TensorRecordItem> t, Chunk& chunk,
    std::map<std::string, TensorPositionInfo>& tensor_position_map) {
  auto t_size = t->size_;
  int64_t prev_offset = 0;

  int64_t best_offset = -1;
  int64_t smallest_gap = std::numeric_limits<int64_t>::max();
  bool success = false;
  chunk.visit([&](Chunk::ChunkNode* x) {
    auto x_size = x->tensor_record_->size_;
    auto x_offset = x->offset_;

    auto max_first_op = std::max(t->start_op_, x->tensor_record_->start_op_);
    auto min_last_op = std::min(t->end_op_, x->tensor_record_->end_op_);
    if (max_first_op <= min_last_op) {
      auto gap = x_offset - prev_offset;

      if (gap >= t_size && gap < smallest_gap) {
        smallest_gap = gap;
        best_offset = prev_offset;
      }
      prev_offset = std::max(prev_offset, x_offset + x_size);
    }
  });

  // the left space of this trunk is enough for this tensor
  if (best_offset == -1 && chunk.size() - prev_offset >= t_size) {
    best_offset = prev_offset;
  }

  if (best_offset == -1) {
    // target tensor can not fit the chunk
    success = false;
  } else {
    success = true;
    chunk.AppendTensor(t, best_offset);
    tensor_position_map.emplace(t->name_,
                                TensorPositionInfo(&chunk, best_offset));
  }

  return success;
}

/*
An offset calculation algorithm designed for variable-length inputs.
 @ params:
 gBertTensorUsageRecord : tensor usage recoders <name, {start_op, end_op, size}>
 global trunk_size_list : a list of list (name, offset)
 @returns:
 assigned_offset : a dict indicates the offset for each tensor
 assigned_trunk : a dict indicates the trunk for each tensor
*/

void ChunkedGreedyBySizeOffsetCalculation(
    const std::vector<TensorRecordItemPtr>& tensor_usage_record,
    ChunkList& chunk_list,
    std::map<std::string, TensorPositionInfo>& tensor_position_map) {
#ifdef NDEBUG
  int64_t new_allocate_size = 0;
#endif
  tensor_position_map.clear();
  chunk_list.Reset();
  // descend order
  for (const auto& t : tensor_usage_record) {
    auto t_name = t->name_;
    auto t_size = t->size_;
    bool is_assigned = false;
    chunk_list.visit([&](Chunk* chunk) {
      if (is_assigned) return;
      is_assigned = TryFitChunk(t, *chunk, tensor_position_map);
    });

    if (!is_assigned) {
      auto new_chunk_size =
          std::max(DEFAULT_TRUNK_SIZE,
                   (static_cast<int64_t>(t_size * K_SCALE) + 31) / 32 * 32);

      Chunk* new_chunk = chunk_list.AddChunk(new_chunk_size);
      new_chunk->AppendTensor(t, 0);
      tensor_position_map.emplace(t_name, TensorPositionInfo(new_chunk, 0));
#ifdef NDEBUG
      new_allocate_size += new_chunk_size;
#endif
    }
  }
  // release not used chunk
  chunk_list.Shrink();
}

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
