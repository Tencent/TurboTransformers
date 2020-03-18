#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "dlpack/dlpack.h"
namespace turbo_transformers {
namespace loaders {

enum class PoolingType {
  kMax = 0,
  kMean,
  kFirst,
  kLast,

};

class BertModel {
 public:
  BertModel(const std::string &filename, DLDeviceType device_type,
            size_t n_layers, int64_t n_heads);
  ~BertModel();

  std::vector<float> operator()(const std::vector<std::vector<int64_t>> &inputs,
                                PoolingType pooling) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m_;
};

}  // namespace loaders
}  // namespace turbo_transformers
