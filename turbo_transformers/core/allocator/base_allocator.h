

#pragma once
#include <string>
#include <vector>

#include "dlpack/dlpack.h"

namespace turbo_transformers {
namespace core {
namespace allocator {

class BaseAllocator {
 public:
  virtual void* allocate(size_t size, DLDeviceType dev,
                         const std::string& name) = 0;
  virtual void free(void* mem, DLDeviceType dev, const std::string& name) = 0;
  // an interface to modify model-aware allocator's model config.
  // the config is encoded in a list of int64_t
  virtual void reset(std::vector<int64_t>& configs){};
  virtual bool is_activation(const std::string& name) const { return false; };
  // TODO(jiaruifang) release all memory cached.
  virtual void release(){};
  virtual ~BaseAllocator();
};

}  // namespace allocator
}  // namespace core
}  // namespace turbo_transformers
