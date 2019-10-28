#include "fast_transformers/core/tensor.h"
#include <memory>
using namespace fastertransformer;

namespace fast_transformers {
namespace kernels {


class Transformer {
public:
  Transformer(const unsigned batch_size, const unsigned  num_layers,
    const unsigned seq_len, const unsigned size_per_head):
    {}
  //how to initialize the kernels
  void init() {
    //init transformer params, who release them?
    for(unsigned i = 0; i < config_.num_layer; i++) {
        std::string prefix = "bert/encoder/layer_" + std::to_string(i) + "/";
        encoder_params_[i].attr_output_layernorm_beta = param_map[prefix + "attention/output/LayerNorm/beta"];
        encoder_params_[i].attr_output_layernorm_gamma = param_map[prefix + "attention/output/LayerNorm/gamma"];
        encoder_params_[i].attr_output_bias = param_map[prefix + "attention/output/dense/bias"];
        encoder_params_[i].attr_output_kernel = param_map[prefix + "attention/output/dense/kernel"];


        encoder_params_[i].attr_bias_K = param_map[prefix + "attention/self/key/bias"];
        encoder_params_[i].attr_kernel_K = param_map[prefix + "attention/self/key/kernel"];
        encoder_params_[i].attr_bias_Q = param_map[prefix + "attention/self/query/bias"];
        encoder_params_[i].attr_kernel_Q = param_map[prefix + "attention/self/query/kernel"];

        encoder_params_[i].attr_bias_V = param_map[prefix + "attention/self/value/bias"];
        encoder_params_[i].attr_kernel_V = param_map[prefix + "attention/self/value/kernel"];
        encoder_params_[i].inter_bias = param_map[prefix + "intermediate/dense/bias"];
        encoder_params_[i].inter_kernel = param_map[prefix + "intermediate/dense/kernel"];

        encoder_params_[i].output_layernorm_beta = param_map[prefix + "output/LayerNorm/beta"];
        encoder_params_[i].output_layernorm_gamma = param_map[prefix + "output/LayerNorm/gamma"];
        encoder_params_[i].output_bias = param_map[prefix + "output/dense/bias"];
        encoder_params_[i].output_kernel = param_map[prefix + "output/dense/kernel"];
    }

    for(int i = 0; i < num_layers_; ++i) {
        encoder_transformer_ops_[i] = make_shared<BertEncoderTransformer<EncoderTraits_, fastertransformer::DeviceType::CPU>>(allocator_, 
                    batch_size_, seq_len_, 
                    seq_len_, head_num_, size_per_head_);
    }
  }
  core::Tensor operator()(const core::Tensor &input_tensor,
                          const core::Tensor &attention_masks) const;

private:
  unsigned batch_size_;
  unsigned num_layers_;
  unsigned seq_len_;
  unsigned head_num_;
  unsigned size_per_head_;

  float* host_malloc(int size, float low = -0.1, float high = 0.1)
  {
    float *ptr = static_cast<float*>(fastertransformer::mkl::wxflow_cpu_alloc(sizeof(float) * size));
    //(*ptr) = (float*)malloc(sizeof(float) * size);
    for(int i = 0; i < size; ++i)
      (*ptr)[i] = low + static_cast <float> (rand()) / static_cast <float> (RAND_MAX / (high - low)); //0.1;
    return ptr;
  }

  std::vector<EncoderInitParam<T, DeviceType::CPU>> encoder_params_;


  BertConfig config_;
  std::shared_ptr<EmbeddingLookupOp<T, DeviceType::CPU>> embedding_lookup_op_;
  std::shared_ptr<EmbeddingPostprocessorOp<T, DeviceType::CPU>> embedding_postprocessor_op_;
  std::vector<std::shared_ptr<BertEncoderTransformer<EncoderTraits_, fastertransformer::DeviceType::CPU>>> encoder_transformer_ops_;
  fastertransformer::Allocator<AllocatorType::CPU> allocator_;
  BertParams<T> bert_params_;
};

} // namespace layers
} // namespace fast_transformers
