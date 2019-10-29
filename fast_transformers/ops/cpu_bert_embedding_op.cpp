#include <vector>
#include <iostream>
#include <stdexcept>
#include "fast_transformers/ops/cpu_bert_embedding_op.h"

namespace fast_transformers {
namespace layers {

template<>
void EmbeddingPostprocessorKernel<float, kDLCPU>(const float* position_embeddings,
                                  const float* segment_embeddings, 
                                  const float* embedding_beta, 
                                  const float* embedding_gamma,
                                  const int* segment_ids,
                                  float* input_tensor,
                                  bool use_segment_embedding,
                                  bool use_position_embedding,
                                  const unsigned token_type_vocab_size,
                                  const unsigned hidden_size,
                                  const unsigned max_position_embeddings,
                                  const unsigned batch_size,
                                  const unsigned seq_length) {
    if(use_segment_embedding) {
        std::vector<float> data(batch_size * seq_length * token_type_vocab_size, 0.);
        for (size_t i = 0; i < batch_size * seq_length; ++i) {
            unsigned id = segment_ids[i];
            if (id >= token_type_vocab_size)
                throw std::out_of_range("lookup_segment_embedding idx error");
            data[i * token_type_vocab_size + id] = 1.;
        }
        std::cout << "before segment_embeddings_ blas" << std::endl;
        
        core::cpu_blas_gemm(false, false,
                            hidden_size, seq_length * batch_size, token_type_vocab_size,
                            1.f,
                            segment_embeddings, hidden_size,
                            &data[0], token_type_vocab_size,
                            1.f,
                            input_tensor, hidden_size);
        
        std::cout << "after segment_embeddings blas" << std::endl;
    }

    if(use_position_embedding) {
        std::cout << "before lookup_position_embedding" << std::endl;
        if(max_position_embeddings < seq_length)
            throw std::out_of_range("lookup_position_embedding idx error");
        for(size_t i = 0; i < batch_size; ++i) {
            for(size_t j = 0; j < seq_length; ++j) {
                for(size_t k = 0; k < hidden_size; ++k) {
                    input_tensor[(i*seq_length + j)*hidden_size + k] += position_embeddings[j*hidden_size + k];
                }  
            }
        }
        std::cout << "after lookup_position_embedding" << std::endl;
    }
    //TODO wait for layernorm_op ready!
    /*
    mkl::layernorm_op(input_tensor, embedding_beta, embedding_gamma,
        batch_size * seq_length, hidden_size);
    */
    std::cout << "after layernorm_op" << std::endl;
    return;
}

template<>
void EmbeddingLookupKernel<float, kDLCPU>(const float* word_embeddings,
                       const int* tokens_ids,
                       float* input_tensor,
                       const unsigned batch_size, 
                       const unsigned seq_length,
                       const unsigned vocab_size,
                       const unsigned hidden_size) {
    std::cout << "before lookup_word_embedding " << batch_size << std::endl;
    //batch * from_seq_length
    for (size_t i = 0; i < batch_size * seq_length; ++i)
    {
        int id = tokens_ids[i];
        if (id >= vocab_size)
            throw std::out_of_range("lookup_word_embedding idx error");
        for(size_t j = 0; j < hidden_size; ++j) {
            *(input_tensor + i * hidden_size + j) = *(word_embeddings + id * hidden_size + j);
        }
    }
    std::cout << "before lookup_word_embedding" << std::endl;
    return;
}

} //layers
} //fast_transformer