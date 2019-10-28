#include "fast_transformers/core/tensor.h"
#include "fast_transformers/dynload/blas.h"
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

PYBIND11_MODULE(fast_transformers, m) {
  m.def("auto_init_blas", &dynload::AutoInitBlas);
  py::class_<core::Tensor>(m, "Tensor")
      .def_static("from_dlpack",
                  [](py::capsule capsule) -> std::unique_ptr<core::Tensor> {
                    auto tensor = (DLManagedTensor *)(capsule);
                    PyCapsule_SetName(capsule.ptr(), "used_tensor");
                    return std::make_unique<core::Tensor>(tensor);
                  })
      .def("to_dlpack",
           [](core::Tensor &tensor) -> py::capsule {
             auto *dlpack = tensor.ToDLPack();
             return py::capsule(dlpack, "dltensor", DLPack_Capsule_Destructor);
           })
      .def("n_dim", &core::Tensor::n_dim)
      .def("shape", &core::Tensor::shape)
      .def("float_data", &core::Tensor::data<float>);
  ;
}

} // namespace python
} // namespace fast_transformers
