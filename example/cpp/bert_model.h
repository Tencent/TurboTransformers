

#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "dlpack/dlpack.h"
#include "turbo_transformers/layers/types.h"

using namespace turbo_transformers;
using PoolType = layers::types::PoolType;

class BertModel {
 public:
  BertModel(const std::string &filename, DLDeviceType device_type,
            size_t n_layers, int64_t n_heads);
  ~BertModel();

  std::vector<float> operator()(
      const std::vector<std::vector<int64_t>> &inputs,
      const std::vector<std::vector<int64_t>> &poistion_ids,
      const std::vector<std::vector<int64_t>> &segment_ids,
      PoolType pooling = PoolType::kFirst, bool use_pooler = false) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};
