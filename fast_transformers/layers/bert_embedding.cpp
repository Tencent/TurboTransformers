#include "fast_transformers/layers/bert_embedding.h"

namespace fast_transformers {
namespace layers {

core::Tensor BERTEmbedding::operator()(const core::Tensor& input_ids, 
                                       const core::Tensor& position_ids, 
                                       const core::Tensor& token_type_ids) const{
    std::cerr << ">>>>>>>>>>>> input_ids <<<<<<<<<<<<" << std::endl;
    input_ids.Print<int>(std::cout);
    std::cerr << ">>>>>>>>>>>> position_ids <<<<<<<<<<<<" << std::endl;
    position_ids.Print<int>(std::cout);
    std::cerr << ">>>>>>>>>>>> token_type_ids <<<<<<<<<<<<" << std::endl;
    token_type_ids.Print<int>(std::cout);

    assert(input_ids.n_dim() == 2);
    const unsigned batch_size = input_ids.shape(0);
    const unsigned seq_length = input_ids.shape(1);
    //TODO 1. switch DeviceType::CPU 2. how should I set stride?
    core::Tensor output_tensor(core::details::CreateDLPackTensor<float, ::fast_transformers::DeviceType::CPU>({batch_size, seq_length, hidden_size_}));

    //embedding_lookup_op_->forward(input_ids.data<int>(), output_tensor.mutableData<float>(), batch_size, seq_length);

    
    EmbeddingLookupKernel<float, kDLCPU>(word_embedings_.data<float>(),
                       input_ids.data<int>(),
                       output_tensor.mutableData<float>(),
                       batch_size, 
                       seq_length,
                       vocab_size_,
                       hidden_size_);
    //embedding_postprocessor_op_->forward(token_type_ids.data<int>(), output_tensor.mutableData<float>(), true, true, batch_size, seq_length);
    
    EmbeddingPostprocessorKernel<float, kDLCPU>(position_embeddings_.data<float>(),
                                 token_type_embeddings_.data<float>(), 
                                 layer_norm_weights_.data<float>(), 
                                 layer_norm_bias_.data<float>(),
                                 token_type_ids.data<int>(),
                                 output_tensor.mutableData<float>(),
                                 true, //use_segment_embedding
                                 true, //use_position_embedding
                                 token_type_vocab_size_,
                                 hidden_size_,
                                 max_position_embeddings_,
                                 batch_size,
                                 seq_length);
                                
    return output_tensor;
}
}
}
