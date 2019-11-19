#include "absl/memory/memory.h"
#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/profiler.h"
#include "fast_transformers/core/tensor.h"
#include "fast_transformers/layers/bert_attention.h"
#include "fast_transformers/layers/bert_embedding.h"
#include "fast_transformers/layers/bert_intermediate.h"
#include "fast_transformers/layers/bert_output.h"
#include "loguru.hpp"
#include "pybind11/pybind11.h"
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

PYBIND11_MODULE(fast_transformers_cxx, m) {
  m.def("set_stderr_verbose_level",
        [](int v) { loguru::g_stderr_verbosity = v; });
  m.def("enable_gperf", &core::EnableGperf);
  m.def("disable_gperf", &core::DisableGperf);

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
      .def("float_data", &core::Tensor::data<float>);

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
      .def("__call__",
           [](layers::BERTEmbedding &self, core::Tensor &input_ids,
              core::Tensor &token_type_ids, core::Tensor &position_ids) {
             core::Tensor output(nullptr);
             self(input_ids, token_type_ids, position_ids, &output);
             return output;
           });

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
      .def("__call__",
           [](layers::BertAttention &self, core::Tensor &input_tensor,
              core::Tensor &attention_mask, core::Tensor &head_mask) {
             core::Tensor output(nullptr);
             self(input_tensor, attention_mask, head_mask, &output);
             return output;
           });

  py::class_<layers::BertIntermediate>(m, "BertIntermediate")
      .def(py::init([](core::Tensor &dense_weight,
                       core::Tensor &dense_bias) -> layers::BertIntermediate * {
        return new layers::BertIntermediate(std::move(dense_weight),
                                            std::move(dense_bias));
      }))
      .def("__call__",
           [](layers::BertIntermediate &self, core::Tensor &input_tensor) {
             core::Tensor output(nullptr);
             self(input_tensor, &output);
             return output;
           });

  py::class_<layers::BertOutput>(m, "BertOutput")
      .def(py::init([](core::Tensor &dense_weight, core::Tensor &dense_bias,
                       core::Tensor &layer_norm_weight,
                       core::Tensor &layer_norm_bias) -> layers::BertOutput * {
        return new layers::BertOutput(
            std::move(dense_weight), std::move(dense_bias),
            std::move(layer_norm_weight), std::move(layer_norm_bias));
      }))
      .def("__call__", [](layers::BertOutput &self, core::Tensor &hidden_states,
                          core::Tensor &input_tensor) {
        core::Tensor output(nullptr);
        self(hidden_states, input_tensor, &output);
        return output;
      });
}

}  // namespace python
}  // namespace fast_transformers
