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

#include "turbo_transformers/loaders/modeling_bert.h"

#include <string>
#include <utility>

#include "cnpy.h"
#include "turbo_transformers/core/tensor_copy.h"
#include "turbo_transformers/layers/bert_attention.h"
#include "turbo_transformers/layers/bert_embedding.h"
#include "turbo_transformers/layers/bert_intermediate.h"
#include "turbo_transformers/layers/bert_output.h"
#include "turbo_transformers/layers/prepare_bert_masks.h"
#include "turbo_transformers/layers/sequence_pool.h"
namespace turbo_transformers {
namespace loaders {

class NPZMapView {
 public:
  NPZMapView(std::string prefix, cnpy::npz_t *npz)
      : prefix_(std::move(prefix)), npz_(npz) {}

  cnpy::NpyArray &operator[](const std::string &key) {
    auto actualKey = prefix_ + key;
    auto it = npz_->find(actualKey);
    TT_ENFORCE(it != npz_->end(), "cannot find parameter %s in npz file",
               actualKey);
    return it->second;
  }

  NPZMapView Sub(const std::string &subview) {
    return NPZMapView(prefix_ + subview + ".", npz_);
  }

 private:
  std::string prefix_;
  cnpy::npz_t *npz_;
};

class NPZLoader {
 public:
  NPZLoader(NPZMapView view, DLDeviceType device)
      : view_(std::move(view)), device_(device) {}

  template <typename T>
  core::Tensor LoadT(const std::string &name) {
    auto &array = view_[name];
    std::vector<int64_t> shape;
    shape.resize(array.shape.size());
    std::copy(array.shape.begin(), array.shape.end(), shape.begin());
    core::Tensor tensor(core::NewDLPackTensorT<T>(shape, device_));
    core::Copy(tensor, array.data<T>(), tensor.numel(), DLDeviceType::kDLCPU);
    return tensor;
  }

  core::Tensor LoadFloat(const std::string &name) { return LoadT<float>(name); }
  core::Tensor operator[](const std::string &name) { return LoadFloat(name); }

 private:
  NPZMapView view_;
  DLDeviceType device_;
};

static std::unique_ptr<layers::BERTEmbedding> LoadEmbedding(NPZMapView npz,
                                                            DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BERTEmbedding>(new layers::BERTEmbedding(
      params["word_embeddings.weight"], params["position_embeddings.weight"],
      params["token_type_embeddings.weight"], params["LayerNorm.weight"],
      params["LayerNorm.bias"], 0));
}

struct BERTLayer {
  explicit BERTLayer(NPZLoader params, int64_t n_heads) {
    attention_.reset(new layers::BertAttention(
        params["attention.qkv.weight"], params["attention.qkv.bias"],
        params["attention.output.dense.weight"],
        params["attention.output.dense.bias"],
        params["attention.output.LayerNorm.weight"],
        params["attention.output.LayerNorm.bias"], n_heads));
    intermediate_.reset(
        new layers::BertIntermediate(params["intermediate.dense.weight"],
                                     params["intermediate.dense.bias"]));
    output_.reset(new layers::BertOutput(
        params["output.dense.weight"], params["output.dense.bias"],
        params["output.LayerNorm.weight"], params["output.LayerNorm.bias"]));
  }

  void operator()(core::Tensor &hidden, core::Tensor &mask,
                  core::Tensor *attention_out, core::Tensor *intermediate_out,
                  core::Tensor *output) {
    (*attention_)(hidden, mask, attention_out);
    (*intermediate_)(*attention_out, intermediate_out);
    (*output_)(*intermediate_out, *attention_out, output);
  }

  std::unique_ptr<layers::BertAttention> attention_;
  std::unique_ptr<layers::BertIntermediate> intermediate_;
  std::unique_ptr<layers::BertOutput> output_;
};

struct BertModel::Impl {
  explicit Impl(const std::string &filename, DLDeviceType device_type,
                size_t n_layers, int64_t n_heads)
      : device_type_(device_type) {
    auto npz = cnpy::npz_load(filename);
    NPZMapView root("", &npz);
    embedding_ = LoadEmbedding(root.Sub("embeddings"), device_type);

    for (size_t i = 0; i < n_layers; ++i) {
      auto view = root.Sub("encoder.layer." + std::to_string(i));
      NPZLoader params(view, device_type);
      encoders_.emplace_back(std::move(params), n_heads);
    }
  }

  std::vector<float> operator()(const std::vector<std::vector<int64_t>> &inputs,
                                PoolingType pooling) {
    int64_t max_seq_len =
        std::accumulate(inputs.begin(), inputs.end(), 0,
                        [](size_t len, const std::vector<int64_t> &input_ids) {
                          return std::max(len, input_ids.size());
                        });
    auto *iptr = inputs_.Reshape<int64_t>(
        {static_cast<int64_t>(inputs.size()), max_seq_len},
        DLDeviceType::kDLCPU, 0);
    auto *mptr = masks_.Reshape<int64_t>(
        {static_cast<int64_t>(inputs.size()), max_seq_len},
        DLDeviceType::kDLCPU, 0);

    for (size_t i = 0; i < inputs.size();
         ++i, iptr += max_seq_len, mptr += max_seq_len) {
      auto &input = inputs[i];
      std::copy(input.begin(), input.end(), iptr);
      std::fill(mptr, mptr + input.size(), 0);
      if (input.size() != static_cast<size_t>(max_seq_len)) {
        std::fill(iptr + input.size(), iptr + max_seq_len, 0);
        std::fill(mptr + input.size(), mptr + max_seq_len, 1);
      }
    }
    if (device_type_ == DLDeviceType::kDLGPU) {
      gpuInputs_.Reshape<int64_t>(
          {static_cast<int64_t>(inputs.size()), max_seq_len},
          DLDeviceType::kDLGPU, 0);
      gpuMasks_.Reshape<int64_t>(
          {static_cast<int64_t>(inputs.size()), max_seq_len},
          DLDeviceType::kDLGPU, 0);
      core::Copy(gpuInputs_, inputs_.data<int64_t>(), inputs_.numel(),
                 DLDeviceType::kDLCPU);
      core::Copy(gpuMasks_, masks_.data<int64_t>(), masks_.numel(),
                 DLDeviceType::kDLCPU);
    }
    core::Tensor seqType(nullptr);
    core::Tensor positionIds(nullptr);
    core::Tensor extendedAttentionMask(nullptr);

    auto &inputIds =
        device_type_ == DLDeviceType::kDLCPU ? inputs_ : gpuInputs_;
    layers::PrepareBertMasks()(
        inputIds, device_type_ == DLDeviceType::kDLCPU ? &masks_ : &gpuMasks_,
        &seqType, &positionIds, &extendedAttentionMask);

    core::Tensor hidden(nullptr);
    (*embedding_)(inputIds, positionIds, seqType, &hidden);
    core::Tensor attOut(nullptr);
    core::Tensor intermediateOut(nullptr);
    for (auto &layer : encoders_) {
      layer(hidden, extendedAttentionMask, &attOut, &intermediateOut, &hidden);
    }
    core::Tensor cpuHidden(nullptr);
    if (hidden.device_type() == DLDeviceType::kDLGPU) {
      cpuHidden.Reshape<float>(
          {hidden.shape(0), hidden.shape(1), hidden.shape(2)},
          DLDeviceType::kDLCPU, 0);
      core::Copy(cpuHidden, hidden.data<float>(), hidden.numel(),
                 hidden.device_type());
    }

    core::Tensor output(nullptr);

    layers::SequencePool(static_cast<layers::types::PoolType>(pooling))(
        hidden.device_type() == DLDeviceType::kDLGPU ? cpuHidden : hidden,
        &output);
    std::vector<float> vec;
    vec.resize(output.numel());
    core::Copy(vec, output);
    return vec;
  }

  std::unique_ptr<layers::BERTEmbedding> embedding_;
  std::vector<BERTLayer> encoders_;
  core::Tensor inputs_{nullptr};
  core::Tensor masks_{nullptr};

  core::Tensor gpuInputs_{nullptr};
  core::Tensor gpuMasks_{nullptr};

  DLDeviceType device_type_;
};

BertModel::BertModel(const std::string &filename, DLDeviceType device_type,
                     size_t n_layers, int64_t n_heads)
    : m_(new Impl(filename, device_type, n_layers, n_heads)) {}
std::vector<float> BertModel::operator()(
    const std::vector<std::vector<int64_t>> &inputs,
    PoolingType pooling) const {
  return m_->operator()(inputs, pooling);
}
BertModel::~BertModel() = default;
}  // namespace loaders
}  // namespace turbo_transformers
