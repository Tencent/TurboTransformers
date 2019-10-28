pybind11_add_module(fast_transformers
        pybind.cpp)

target_link_libraries(fast_transformers PRIVATE bert_embedding)
