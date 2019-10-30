#pragma once
#include <vector>
#include <cstring>
#include <iostream>

namespace fast_transformers {
namespace ops {

/***
 * Trans from (B, S, N, H) -> (B, N, S, H)
 * **/
template <class T>
void cpu_transpose_for_score_long(const T* a, const std::vector<unsigned>& shape, T* b);

template <class T>
void cpu_add_bias_transpose_for_score_long(const T* a,
    const T* bias, const std::vector<unsigned>& shape, T* b);


template <class T>
void cpu_transpose_for_score_short(const T* a, const std::vector<unsigned>& shape, T* b);

template <class T>
void cpu_add_bias(const T* bias,
    T* data,
    const int M,
    const int N);

template<typename T>
void cpu_add_bias_input_layernorm_op(T* out, const T* input, const T* bias, 
  const T* gamma, const T* beta, int m, int n);

template<typename T>
void cpu_layernorm_op(T* input, 
    const T* gamma, const T* beta, const int m, const int n);


} //namespace fast_transformers
} //namespace ops