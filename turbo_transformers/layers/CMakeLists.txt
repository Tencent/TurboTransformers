

add_subdirectory(kernels)

add_library(tt_layers OBJECT
        bert_embedding.cpp
        bert_attention.cpp
        bert_intermediate.cpp
        bert_output.cpp
        sequence_pool.cpp
        bert_pooler.cpp
        prepare_bert_masks.cpp
        albert_layer.cpp
        multi_headed_attention.cpp
        positionwise_ffn.cpp
        addbias_act.cpp
        addbias_layernorm.cpp
        )

target_link_libraries(tt_layers PUBLIC tt_core tt_kernels)

add_executable(tt_layers_test prepare_bert_masks_test.cpp)
target_link_libraries(tt_layers_test catch2_test_main tt_layers tt_core tt_kernels)
