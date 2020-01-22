#ifdef FT_WITH_CUDA
#include "fast_transformers/core/cuda_error.h"
#endif
#include "fast_transformers/core/memory.h"

namespace fast_transformers {
namespace test {

using Tensor = fast_transformers::core::Tensor;

#ifdef FT_WITH_CUDA
template <typename T>
void FillDataForCPUGPUTensors(Tensor& cpu_tensor, Tensor& gpu_tensor) {
  T* gpu_data = gpu_tensor.mutableData<T>();
  T* cpu_data = cpu_tensor.mutableData<T>();
  auto size = cpu_tensor.numel();
  srand((unsigned)time(NULL));
  for (int64_t i = 0; i < size; ++i) {
    cpu_data[i] = rand() / static_cast<T>(RAND_MAX);
  }
  fast_transformers::core::FT_Memcpy(
      gpu_data, cpu_data, size * sizeof(T),
      ::fast_transformers::core::MemcpyFlag::kCPU2GPU);
}

template <typename T>
bool CompareCPUGPU(const Tensor& cpu_tensor, const Tensor& gpu_tensor) {
  const T* gpu_data = gpu_tensor.data<T>();
  const T* cpu_data = cpu_tensor.data<T>();
  auto size = cpu_tensor.numel();

  std::unique_ptr<T[]> gpu_data_ref(new T[size]);
  fast_transformers::core::FT_Memcpy(
      gpu_data_ref.get(), gpu_data, size * sizeof(T),
      fast_transformers::core::MemcpyFlag::kGPU2CPU);
  bool ret = true;
  for (int64_t i = 0; i < size; ++i) {
    if (std::abs(gpu_data_ref[i] - cpu_data[i]) > 1e-3) {
      std::cerr << "@ " << i << ": " << gpu_data_ref[i] << " vs " << cpu_data[i]
                << std::endl;
      ret = false;
      break;
    }
  }
  return ret;
}
#endif

inline void RandomFillHost(float* m, const int mSize, float LO = 0.,
                           float HI = 1.) {
  srand(static_cast<unsigned>(time(0)));
  for (int i = 0; i < mSize; i++)
    m[i] = LO + static_cast<float>(rand()) /
                    (static_cast<float>(RAND_MAX / (HI - LO)));
}

template <typename T>
void Fill(fast_transformers::core::Tensor& tensor, T lower_bound = 0.,
          T upper_bound = 1.) {
  T* T_data = tensor.mutableData<T>();
  auto size = tensor.numel();
  std::unique_ptr<T> cpu_data(new T[size]);
  RandomFillHost(cpu_data.get(), size, lower_bound, upper_bound);

  fast_transformers::core::MemcpyFlag flag;
  if (tensor.device_type() == kDLCPU) {
    flag = fast_transformers::core::MemcpyFlag::kCPU2CPU;
  } else if (tensor.device_type() == kDLGPU) {
    flag = fast_transformers::core::MemcpyFlag::kCPU2GPU;
  } else {
    FT_THROW("Fill device_type wrong");
  }
  ::fast_transformers::core::FT_Memcpy(T_data, cpu_data.get(), size * sizeof(T),
                                       flag);
}

}  // namespace test
}  // namespace fast_transformers
