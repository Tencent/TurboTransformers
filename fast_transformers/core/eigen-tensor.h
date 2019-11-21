#pragma once
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"
namespace fast_transformers {
namespace core {

template <int Order>
using EigenFloatTensor = Eigen::TensorMap<Eigen::Tensor<float, Order>>;

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

template <int Order>
inline EigenFloatTensor<Order> to_tensor(Tensor* t);
template <int Order>
inline const EigenFloatTensor<Order> to_tensor(const Tensor& t) {
  auto* t_ptr = const_cast<Tensor*>(&t);
  return to_tensor<Order>(t_ptr);
}

template <>
inline EigenFloatTensor<0> to_tensor<0>(Tensor* t) {
  FT_ENFORCE_EQ(t->numel(), 1, "0 dim tensor must contains 1 elems");
  return EigenFloatTensor<0>(t->mutableData<float>());
}

template <>
inline EigenFloatTensor<1> to_tensor<1>(Tensor* t) {
  return EigenFloatTensor<1>(t->mutableData<float>(), (int)t->numel());
}

template <>
inline EigenFloatTensor<2> to_tensor<2>(Tensor* t) {
  FT_ENFORCE_EQ(t->n_dim(), 2, "must be matrix");
  return EigenFloatTensor<2>(t->mutableData<float>(), (int)t->shape(0),
                             (int)t->shape(1));
}

template <>
inline EigenFloatTensor<3> to_tensor<3>(Tensor* t) {
  FT_ENFORCE_EQ(t->n_dim(), 3, "n_dim should be 3");
  return EigenFloatTensor<3>(t->mutableData<float>(), (int)t->shape(0),
                             (int)t->shape(1), (int)t->shape(2));
}

template <>
inline EigenFloatTensor<4> to_tensor<4>(Tensor* t) {
  FT_ENFORCE_EQ(t->n_dim(), 4, "n_dim should be 4");
  return EigenFloatTensor<4>(t->mutableData<float>(), (int)t->shape(0),
                             (int)t->shape(1), (int)t->shape(2),
                             (int)t->shape(3));
}

extern Eigen::ThreadPoolDevice& CPUDevice();

}  // namespace core
}  // namespace fast_transformers
