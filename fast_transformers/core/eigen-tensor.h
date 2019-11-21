#pragma once
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"
namespace fast_transformers {
namespace core {

template <int Order, typename T>
using EigenTensor = Eigen::TensorMap<Eigen::Tensor<T, Order>>;

using Vector =
    Eigen::Matrix<float, Eigen::Dynamic,
                  1>;  // for vector, row major and col major are same.
using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline Eigen::Map<Vector, Eigen::Aligned64> to_vector(Tensor* t) {
  return Eigen::Map<Vector, Eigen::Aligned64>(t->mutableData<float>(),
                                              t->numel());
}

inline const Eigen::Map<Vector, Eigen::Aligned64> to_vector(const Tensor& t) {
  return to_vector(const_cast<Tensor*>(&t));
}

inline Eigen::Map<Matrix, Eigen::Aligned64> to_mat(Tensor* t) {
  return Eigen::Map<Matrix, Eigen::Aligned64>(t->mutableData<float>(),
                                              t->rows(), t->cols());
}

inline const Eigen::Map<Matrix, Eigen::Aligned64> to_mat(const Tensor& t) {
  return to_mat(const_cast<Tensor*>(&t));
}

namespace details {
template <int Order, typename T>
struct ToTensorImpl {
  EigenTensor<Order, T> operator()(Tensor* t) const;
};

template <typename T>
struct ToTensorImpl<0, T> {
  EigenTensor<0, T> operator()(Tensor* t) const {
    FT_ENFORCE_EQ(t->numel(), 1, "0 dim tensor must contains 1 elems");
    return EigenTensor<0, T>(t->mutableData<T>());
  }
};

template <typename T>
struct ToTensorImpl<1, T> {
  inline EigenTensor<1, T> operator()(Tensor* t) {
    return EigenTensor<1, T>(t->mutableData<T>(), (int)t->numel());
  }
};

template <typename T>
struct ToTensorImpl<2, T> {
  inline EigenTensor<2, T> operator()(Tensor* t) {
    FT_ENFORCE_EQ(t->n_dim(), 2, "must be matrix");
    return EigenTensor<2, T>(t->mutableData<T>(), (int)t->shape(0),
                             (int)t->shape(1));
  }
};
template <typename T>
struct ToTensorImpl<3, T> {
  inline EigenTensor<3, T> operator()(Tensor* t) {
    FT_ENFORCE_EQ(t->n_dim(), 3, "n_dim should be 3");
    return EigenTensor<3, T>(t->mutableData<T>(), (int)t->shape(0),
                             (int)t->shape(1), (int)t->shape(2));
  }
};

template <typename T>
struct ToTensorImpl<4, T> {
  inline EigenTensor<4, T> operator()(Tensor* t) {
    FT_ENFORCE_EQ(t->n_dim(), 4, "n_dim should be 4");
    return EigenTensor<4, T>(t->mutableData<T>(), (int)t->shape(0),
                             (int)t->shape(1), (int)t->shape(2),
                             (int)t->shape(3));
  }
};

}  // namespace details

template <int Order, typename T>
inline EigenTensor<Order, T> to_tensor(Tensor* t) {
  return details::ToTensorImpl<Order, T>()(t);
}
template <int Order, typename T>
inline const EigenTensor<Order, T> to_tensor(const Tensor& t) {
  auto* t_ptr = const_cast<Tensor*>(&t);
  return to_tensor<Order, T>(t_ptr);
}

extern Eigen::ThreadPoolDevice& CPUDevice();

}  // namespace core
}  // namespace fast_transformers
