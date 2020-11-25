

add_library(bert_model bert_model.cpp)
target_link_libraries(bert_model
        PUBLIC tt_npz_loader
        PRIVATE tt_layers tt_kernels)

add_executable(bert_model_example bert_model_example.cpp)
target_link_libraries(bert_model_example bert_model)
