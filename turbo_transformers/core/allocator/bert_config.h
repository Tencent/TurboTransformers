

#pragma once
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "turbo_transformers/core/allocator/model_aware_memory_scheduler.h"

namespace turbo_transformers {
namespace core {
namespace allocator {
namespace bert_config {

template <typename T>
void GetBertTensorUsageRecord(
    std::vector<TensorRecordItemPtr>& TensorUsageRecord,
    std::set<std::string>& activation_set, int64_t batch_size, int64_t seq_len,
    int64_t num_head = 12, int64_t hidden_size = 768, int64_t num_layer = 12);

}  // namespace bert_config
}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
