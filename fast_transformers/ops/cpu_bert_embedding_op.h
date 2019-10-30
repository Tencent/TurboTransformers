#pragma once
#include "fast_transformers/core/math_function.h"
#include "fast_transformers/ops/cpu_layernorm_op.h"
#include "fast_transformers/ops/ops_interfaces.h"

namespace fast_transformers {
namespace ops {

//TODO please choose either a function or a class.
//I prefer a function, because it can be split to a .cpp file, while it has too many args.
template<typename T, DLDeviceType Device>
void EmbeddingPostprocessorKernel(const T* position_embeddings,
                                  const T* segment_embeddings, 
                                  const T* embedding_beta, 
                                  const T* embedding_gamma,
                                  const int* segment_ids,
                                  T* input_tensor,
                                  bool use_segment_embedding,
                                  bool use_position_embedding,
                                  const unsigned token_type_vocab_size,
                                  const unsigned hidden_size,
                                  const unsigned max_position_embeddings,
                                  const unsigned batch_size,
                                  const unsigned seq_length);

template<typename T, DLDeviceType Device>
void EmbeddingLookupKernel(const T* word_embeddings,
                       const int* tokens_ids,
                       T* input_tensor,
                       const unsigned batch_size, 
                       const unsigned seq_length,
                       const unsigned vocab_size,
                       const unsigned hidden_size);

template<typename T>
class EmbeddingPostprocessorOp<T, kDLCPU>{
public:
    EmbeddingPostprocessorOp(const unsigned token_type_vocab_size, 
                            const unsigned hidden_size, 
                            const unsigned max_position_embeddings):token_type_vocab_size_(token_type_vocab_size),
                            hidden_size_(hidden_size),
                            max_position_embeddings_(max_position_embeddings) {}

    void initialize(const T* position_embeddings, const T* segment_embeddings, const T* embedding_beta, const T* embedding_gamma){
        position_embeddings_ = position_embeddings;
        segment_embeddings_ = segment_embeddings;
        embedding_beta_ = embedding_beta;
        embedding_gamma_ = embedding_gamma;
    }

    void forward(const int* segment_ids,
            T* input_tensor,
            bool use_segment_embedding,
            bool use_position_embedding,
            const unsigned batch_size,
            const unsigned seq_length) const {
        if(use_segment_embedding) {
            std::vector<T> data(batch_size * seq_length * token_type_vocab_size_, 0.);
            for (size_t i = 0; i < batch_size * seq_length; ++i) {
                unsigned id = segment_ids[i];
                if (id >= token_type_vocab_size_)
                    throw std::out_of_range("lookup_segment_embedding idx error");
                data[i * token_type_vocab_size_ + id] = 1.;
            }
            std::cout << "before segment_embeddings_ blas" << std::endl;
           
            core::cpu_blas_gemm(false, false,
                                hidden_size_, seq_length * batch_size, token_type_vocab_size_,
                                1.f,
                                segment_embeddings_, hidden_size_,
                                &data[0], token_type_vocab_size_,
                                1.f,
                                input_tensor, hidden_size_);
            
            std::cout << "after segment_embeddings_ blas" << std::endl;
        }
        
        if(use_position_embedding) {
            std::cout << "before lookup_position_embedding" << std::endl;
            if(max_position_embeddings_ < seq_length)
                throw std::out_of_range("lookup_position_embedding idx error");
            for(size_t i = 0; i < batch_size; ++i) {
                for(size_t j = 0; j < seq_length; ++j) {
                    for(size_t k = 0; k < hidden_size_; ++k) {
                        input_tensor[(i*seq_length + j)*hidden_size_ + k] += position_embeddings_[j*hidden_size_ + k];
                    }  
                }
            }
            std::cout << "after lookup_position_embedding" << std::endl;
        }
        ops::cpu_layernorm_op(input_tensor, embedding_beta_, embedding_gamma_,
            batch_size * seq_length, hidden_size_);
        std::cout << "after layernorm_op" << std::endl;
        return;
    }
private:
    unsigned token_type_vocab_size_, hidden_size_;
    unsigned max_position_embeddings_;
    const T* embedding_beta_, *embedding_gamma_;
    const T* position_embeddings_;
    const T* segment_embeddings_;
};

//TODO jiaruifang can not ensure the correctness of this class
template<typename T>
class EmbeddingLookupOp<T, kDLCPU> {
public:
    EmbeddingLookupOp(const unsigned vocab_size, const unsigned hidden_size):vocab_size_(vocab_size),hidden_size_(hidden_size) {}
    void initialize(const T* word_embeddings) {
        word_embeddings_ = word_embeddings;
    }
    void forward(const int* tokens_ids, T* input_tensor, const unsigned batch_size, const unsigned seq_length) const {
        std::cout << "before lookup_word_embedding " << batch_size << std::endl;
        //batch * from_seq_length
        for (size_t i = 0; i < batch_size * seq_length; ++i)
        {
            unsigned id = tokens_ids[i];
            if (id >= vocab_size_)
                throw std::out_of_range("lookup_word_embedding idx error");
            for(size_t j = 0; j < hidden_size_; ++j) {
                *(input_tensor + i * hidden_size_ + j) = *(word_embeddings_ + id * hidden_size_ + j);
            }
        }
        std::cout << "before lookup_word_embedding" << std::endl;
        return;
    }

private:
    const T* word_embeddings_;
    unsigned vocab_size_, hidden_size_;
};

} //layers
} //fast_transformer
