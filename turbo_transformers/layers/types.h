

#pragma once
namespace turbo_transformers {
namespace layers {
namespace types {

enum class ReduceType { kMax = 0, kSum };
enum class ActivationType { Gelu = 0, Tanh = 1, Relu = 2 };
enum class PoolType { kMax = 0, kMean, kFirst, kLast };
}  // namespace types
}  // namespace layers
}  // namespace turbo_transformers
