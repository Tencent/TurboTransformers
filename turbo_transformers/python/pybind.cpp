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

#include <pybind11/stl.h>

#include "absl/memory/memory.h"
#include "loguru.hpp"
#include "pybind11/pybind11.h"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/config.h"
#include "turbo_transformers/core/model_aware_allocator.h"
#include "turbo_transformers/core/profiler.h"
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/addbias_act.h"
#include "turbo_transformers/layers/addbias_layernorm.h"
#include "turbo_transformers/layers/albert_layer.h"
#include "turbo_transformers/layers/bert_attention.h"
#include "turbo_transformers/layers/bert_embedding.h"
#include "turbo_transformers/layers/bert_intermediate.h"
#include "turbo_transformers/layers/bert_output.h"
#include "turbo_transformers/layers/bert_pooler.h"
#include "turbo_transformers/layers/multi_headed_attention.h"
#include "turbo_transformers/layers/positionwise_ffn.h"
#include "turbo_transformers/layers/prepare_bert_masks.h"
#include "turbo_transformers/layers/sequence_pool.h"

namespace turbo_transformers {
namespace python {

namespace py = pybind11;

static void DLPack_Capsule_Destructor(PyObject *data) {
  auto *dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(data, "dltensor");
  if (dlMTensor) {
    // the dlMTensor has not been consumed, call deleter ourselves
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    dlMTensor->deleter(const_cast<DLManagedTensor *>(dlMTensor));
  } else {
    // the dlMTensor has been consumed
    // PyCapsule_GetPointer has set an error indicator
    PyErr_Clear();
  }
}

static void BindConfig(py::module &m) {
  py::enum_<core::BlasProvider>(m, "BlasProvider")
      .value("MKL", core::BlasProvider::MKL)
      .value("OpenBlas", core::BlasProvider::OpenBlas);

  m.def("is_compiled_with_cuda", &core::IsCompiledWithCUDA)
      .def("get_blas_provider", &core::GetBlasProvider);
}

PYBIND11_MODULE(turbo_transformers_cxx, m) {
  char *argv[] = {strdup("turbo_transformers_cxx"), nullptr};
  int argc = 1;
  loguru::init(argc, argv);

  auto config_module =
      m.def_submodule("config", "compile configuration of turbo_transformers");

  BindConfig(config_module);

  m.def("set_stderr_verbose_level",
        [](int v) { loguru::g_stderr_verbosity = v; });
  m.def("enable_perf", &core::EnableGperf);
  m.def("disable_perf", &core::DisableGperf);
  m.def("set_num_threads", &core::SetNumThreads);

  m.def("mem_reserve", &core::reserve_api);
  m.def("static_mem_schedule", &core::static_schedule_api);
  m.def("dynamic_mem_schedule", &core::schedule_dynamic_api);

  py::class_<core::Tensor>(m, "Tensor")
      .def_static("from_dlpack",
                  [](py::capsule capsule) -> std::unique_ptr<core::Tensor> {
                    auto tensor = (DLManagedTensor *)(capsule);
                    PyCapsule_SetName(capsule.ptr(), "used_tensor");
                    return absl::make_unique<core::Tensor>(tensor);
                  })
      .def("to_dlpack",
           [](core::Tensor &tensor) -> py::capsule {
             auto *dlpack = tensor.ToDLPack();
             return py::capsule(dlpack, "dltensor", DLPack_Capsule_Destructor);
           })
      .def("n_dim", &core::Tensor::n_dim)
      .def("shape", &core::Tensor::shape)
      .def("float_data", &core::Tensor::data<float>)
      .def_static("create_empty", [] { return core::Tensor(nullptr); });

  py::class_<layers::BERTEmbedding>(m, "BERTEmbedding")
      .def(py::init(
          [](core::Tensor &word_embeddings, core::Tensor &position_embeddings,
             core::Tensor &token_type_embeddings,
             core::Tensor &layer_norm_weights,
             core::Tensor &layer_norm_bias) -> layers::BERTEmbedding * {
            return new layers::BERTEmbedding(
                std::move(word_embeddings), std::move(position_embeddings),
                std::move(token_type_embeddings), std::move(layer_norm_weights),
                std::move(layer_norm_bias));
          }))
      .def("__call__", &layers::BERTEmbedding::operator());

  py::class_<layers::BertAttention>(m, "BertAttention")
      .def(py::init([](core::Tensor &qkv_weight, core::Tensor &qkv_bias,
                       core::Tensor &dense_weight, core::Tensor &dense_bias,
                       core::Tensor &layer_norm_weight,
                       core::Tensor &layer_norm_bias,
                       int num_attention_heads) -> layers::BertAttention * {
        return new layers::BertAttention(
            std::move(qkv_weight), std::move(qkv_bias), std::move(dense_weight),
            std::move(dense_bias), std::move(layer_norm_weight),
            std::move(layer_norm_bias), num_attention_heads);
      }))
      .def("__call__", &layers::BertAttention::operator());

  py::class_<layers::MultiHeadedAttention>(m, "MultiHeadedAttention")
      .def(py::init(
          [](core::Tensor &key_weight, core::Tensor &key_bias,
             core::Tensor &value_weight, core::Tensor &value_bias,
             core::Tensor &query_weight, core::Tensor &query_bias,
             core::Tensor &dense_weight, core::Tensor &dense_bias,
             core::Tensor &qkv_weight, core::Tensor &qkv_bias,
             int num_attention_heads) -> layers::MultiHeadedAttention * {
            return new layers::MultiHeadedAttention(
                std::move(key_weight), std::move(key_bias),
                std::move(value_weight), std::move(value_bias),
                std::move(query_weight), std::move(query_bias),
                std::move(dense_weight), std::move(dense_bias),
                std::move(qkv_weight), std::move(qkv_bias),
                num_attention_heads);
          }))
      .def(py::init(
          [](core::Tensor &key_weight, core::Tensor &key_bias,
             core::Tensor &value_weight, core::Tensor &value_bias,
             core::Tensor &query_weight, core::Tensor &query_bias,
             core::Tensor &dense_weight, core::Tensor &dense_bias,
             core::Tensor &qkv_weight, core::Tensor &qkv_bias,
             core::Tensor &layernorm_gamma, core::Tensor &layernorm_beta,
             int num_attention_heads) -> layers::MultiHeadedAttention * {
            return new layers::MultiHeadedAttention(
                std::move(key_weight), std::move(key_bias),
                std::move(value_weight), std::move(value_bias),
                std::move(query_weight), std::move(query_bias),
                std::move(dense_weight), std::move(dense_bias),
                std::move(qkv_weight), std::move(qkv_bias),
                std::move(layernorm_gamma), std::move(layernorm_beta),
                num_attention_heads);
          }))
      .def("__call__", &layers::MultiHeadedAttention::operator());

  py::class_<layers::BertIntermediate>(m, "BertIntermediate")
      .def(py::init([](core::Tensor &dense_weight,
                       core::Tensor &dense_bias) -> layers::BertIntermediate * {
        return new layers::BertIntermediate(std::move(dense_weight),
                                            std::move(dense_bias));
      }))
      .def("__call__", &layers::BertIntermediate::operator());

  py::class_<layers::BertOutput>(m, "BertOutput")
      .def(py::init([](core::Tensor &dense_weight, core::Tensor &dense_bias,
                       core::Tensor &layer_norm_weight,
                       core::Tensor &layer_norm_bias) -> layers::BertOutput * {
        return new layers::BertOutput(
            std::move(dense_weight), std::move(dense_bias),
            std::move(layer_norm_weight), std::move(layer_norm_bias));
      }))
      .def("__call__", &layers::BertOutput::operator());

  py::class_<layers::SequencePool>(m, "SequencePool")
      .def(py::init([](const std::string &pool_type) -> layers::SequencePool * {
        return new layers::SequencePool(pool_type);
      }))
      .def("__call__", &layers::SequencePool::operator());

  py::class_<layers::BertPooler>(m, "BertPooler")
      .def(py::init([](core::Tensor &dense_weight,
                       core::Tensor &dense_bias) -> layers::BertPooler * {
        return new layers::BertPooler(std::move(dense_weight),
                                      std::move(dense_bias));
      }))
      .def("__call__", &layers::BertPooler::operator());

  py::class_<layers::PrepareBertMasks>(m, "PrepareBertMasks")
      .def(py::init())
      .def("__call__", &layers::PrepareBertMasks::operator());

  py::class_<layers::AlbertLayer>(m, "AlbertLayer")
      .def(py::init([](core::Tensor &dense_weight, core::Tensor &dense_bias,
                       core::Tensor &dense_output_weight,
                       core::Tensor &dense_output_bias,
                       core::Tensor &layer_norm_weight,
                       core::Tensor &layer_norm_bias) -> layers::AlbertLayer * {
        return new layers::AlbertLayer(
            std::move(dense_weight), std::move(dense_bias),
            std::move(dense_output_weight), std::move(dense_output_bias),
            std::move(layer_norm_weight), std::move(layer_norm_bias));
      }))
      .def("__call__", &layers::AlbertLayer::operator());

  py::class_<layers::PositionwiseFeedForward>(m, "PositionwiseFeedForward")
      .def(py::init([](core::Tensor &dense_weight_1, core::Tensor &dense_bias_1,
                       core::Tensor &dense_weight_2, core::Tensor &dense_bias_2,
                       core::Tensor &layer_norm_weight,
                       core::Tensor &layer_norm_bias)
                        -> layers::PositionwiseFeedForward * {
        return new layers::PositionwiseFeedForward(
            std::move(dense_weight_1), std::move(dense_bias_1),
            std::move(dense_weight_2), std::move(dense_bias_2),
            std::move(layer_norm_weight), std::move(layer_norm_bias));
      }))
      .def("__call__", &layers::PositionwiseFeedForward::operator());

  py::class_<layers::FusedAddBiasGELU>(m, "FusedAddBiasGELU")
      .def(py::init([](core::Tensor &dense_bias) -> layers::FusedAddBiasGELU * {
        return new layers::FusedAddBiasGELU(std::move(dense_bias));
      }))
      .def("__call__", &layers::FusedAddBiasGELU::operator());

  py::class_<layers::FusedAddBiasLayerNorm>(m, "FusedAddBiasLayerNorm")
      .def(py::init(
          [](core::Tensor &dense_bias, core::Tensor &layer_norm_weight,
             core::Tensor &layer_norm_bias) -> layers::FusedAddBiasLayerNorm * {
            return new layers::FusedAddBiasLayerNorm(
                std::move(dense_bias), std::move(layer_norm_weight),
                std::move(layer_norm_bias));
          }))
      .def("__call__", &layers::FusedAddBiasLayerNorm::operator());
}

}  // namespace python
}  // namespace turbo_transformers
