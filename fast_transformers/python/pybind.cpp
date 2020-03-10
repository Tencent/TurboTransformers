// Copyright 2020 Tencent
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/memory/memory.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/config.h"
#include "fast_transformers/core/profiler.h"
#include "fast_transformers/core/tensor.h"
#include "fast_transformers/layers/bert_attention.h"
#include "fast_transformers/layers/bert_embedding.h"
#include "fast_transformers/layers/bert_intermediate.h"
#include "fast_transformers/layers/bert_output.h"
#include "fast_transformers/layers/prepare_bert_masks.h"
#include "fast_transformers/layers/sequence_pool.h"
#include "loguru.hpp"
#include "pybind11/pybind11.h"
#ifdef _OPENMP
#include "omp.h"
#endif

namespace fast_transformers {
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
  m.def("is_with_cuda", core::IsWithCUDA)
      .def("get_blas_provider", core::GetBlasProvider);
}

PYBIND11_MODULE(fast_transformers_cxx, m) {
  char *argv[] = {strdup("fast_transformers_cxx"), nullptr};
  int argc = 1;
  loguru::init(argc, argv);

  auto config_module =
      m.def_submodule("config", "compile configuration of fast transformers");

  BindConfig(config_module);

  m.def("set_stderr_verbose_level",
        [](int v) { loguru::g_stderr_verbosity = v; });
  m.def("enable_gperf", &core::EnableGperf);
  m.def("disable_gperf", &core::DisableGperf);
  m.def("set_num_threads", [](int n_th) {
  // The order seems important. Set MKL NUM_THREADS before OMP.
#ifdef FT_BLAS_USE_MKL
    mkl_set_num_threads(n_th);
#else
    openblas_set_num_threads(n_th);
#endif
#ifdef _OPENMP
    omp_set_num_threads(n_th);
#endif
  });
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
             core::Tensor &layer_norm_weights, core::Tensor &layer_norm_bias,
             float dropout_rate) -> layers::BERTEmbedding * {
            return new layers::BERTEmbedding(
                std::move(word_embeddings), std::move(position_embeddings),
                std::move(token_type_embeddings), std::move(layer_norm_weights),
                std::move(layer_norm_bias), dropout_rate);
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

  py::class_<layers::PrepareBertMasks>(m, "PrepareBertMasks")
      .def(py::init())
      .def("__call__", &layers::PrepareBertMasks::operator());
}

}  // namespace python
}  // namespace fast_transformers
