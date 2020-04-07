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
#include "turbo_transformers/layers/bert_pooler.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/prepare_bert_masks.h"
#include "turbo_transformers/layers/sequence_pool.h"
#include "turbo_transformers/loaders/npz_load.h"
namespace turbo_transformers {
namespace loaders {

static std::unique_ptr<layers::BERTEmbedding> LoadEmbedding(NPZMapView npz,
                                                            DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BERTEmbedding>(new layers::BERTEmbedding(
      params["word_embeddings.weight"], params["position_embeddings.weight"],
      params["token_type_embeddings.weight"], params["LayerNorm.weight"],
      params["LayerNorm.bias"], 0));
}

static std::unique_ptr<layers::BertPooler> LoadPooler(NPZMapView npz,
                                                      DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BertPooler>(
      new layers::BertPooler(params["dense.weight"], params["dense.bias"]));
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

    if (root.IsExist("pooler")) {
      pooler_ = LoadPooler(root.Sub("pooler"), device_type);
    }
  }

  template <typename T>
  void PadTensor(const std::vector<std::vector<T>> &data_array, int64_t n,
                 int64_t m, T pad_val, DLDeviceType device_type,
                 core::Tensor *output_tensor) {
    if (m == 0 || n == 0 || data_array.size() == 0) {
      return;
    }
    core::Tensor cpu_tensor(nullptr);
    T *tensor_data_ptr;
    if (device_type == DLDeviceType::kDLGPU) {
      tensor_data_ptr = cpu_tensor.Reshape<T>({n, m}, DLDeviceType::kDLCPU, 0);
      output_tensor->Reshape<T>({n, m}, device_type, 0);
    } else {
      tensor_data_ptr = output_tensor->Reshape<T>({n, m}, device_type, 0);
    }
    for (int64_t i = 0; i < n; ++i, tensor_data_ptr += m) {
      auto &line = data_array[i];
      if (line.size() > 0) {
        core::Copy(line.data(), line.size(), DLDeviceType::kDLCPU,
                   DLDeviceType::kDLCPU, tensor_data_ptr);
      }
      if (line.size() != static_cast<size_t>(m)) {
        layers::kernels::common::Fill(tensor_data_ptr + line.size(),
                                      static_cast<size_t>(m) - line.size(),
                                      pad_val, DLDeviceType::kDLCPU);
      }
    }
    if (device_type == DLDeviceType::kDLGPU) {
      core::Copy<T>(cpu_tensor, *output_tensor);
    }
  }

  std::vector<float> operator()(
      const std::vector<std::vector<int64_t>> &inputs,
      const std::vector<std::vector<int64_t>> &poistion_ids,
      const std::vector<std::vector<int64_t>> &segment_ids, PoolType pooling,

      bool use_pooler) {
    core::Tensor inputs_{nullptr};
    core::Tensor masks_{nullptr};

    core::Tensor gpuInputs_{nullptr};
    core::Tensor gpuMasks_{nullptr};

    int64_t max_seq_len =
        std::accumulate(inputs.begin(), inputs.end(), 0,
                        [](size_t len, const std::vector<int64_t> &input_ids) {
                          return std::max(len, input_ids.size());
                        });
    int64_t batch_size = inputs.size();
    auto *iptr = inputs_.Reshape<int64_t>({batch_size, max_seq_len},
                                          DLDeviceType::kDLCPU, 0);
    auto *mptr = masks_.Reshape<int64_t>({batch_size, max_seq_len},
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
      gpuInputs_.Reshape<int64_t>({batch_size, max_seq_len},
                                  DLDeviceType::kDLGPU, 0);
      gpuMasks_.Reshape<int64_t>({batch_size, max_seq_len},
                                 DLDeviceType::kDLGPU, 0);
      core::Copy(inputs_.data<int64_t>(), inputs_.numel(), DLDeviceType::kDLCPU,
                 gpuInputs_);
      core::Copy(masks_.data<int64_t>(), masks_.numel(), DLDeviceType::kDLCPU,
                 gpuMasks_);
    }
    auto &inputIds =
        device_type_ == DLDeviceType::kDLCPU ? inputs_ : gpuInputs_;

    core::Tensor seqType(nullptr);
    core::Tensor positionIds(nullptr);
    core::Tensor extendedAttentionMask(nullptr);
    if (poistion_ids.size() != 0) {
      TT_ENFORCE_EQ(
          poistion_ids.size(), static_cast<size_t>(batch_size),
          "Position ids should have the same batch size as ibout ids");
      PadTensor(poistion_ids, batch_size, max_seq_len, static_cast<int64_t>(0),
                device_type_, &positionIds);
    }
    if (segment_ids.size() != 0) {
      TT_ENFORCE_EQ(segment_ids.size(), static_cast<size_t>(batch_size),
                    "Segment ids should have the same batch size as ibout ids");
      PadTensor(segment_ids, batch_size, max_seq_len, static_cast<int64_t>(0),
                device_type_, &seqType);
    }

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

    core::Tensor poolingOutput(nullptr);
    layers::SequencePool(static_cast<layers::types::PoolType>(pooling))(
        hidden, &poolingOutput);
    std::vector<float> vec;
    if (use_pooler) {
      core::Tensor output(nullptr);
      (*pooler_)(poolingOutput, &output);
      vec.resize(output.numel());
      core::Copy(output, vec);
    } else {
      vec.resize(poolingOutput.numel());
      core::Copy(poolingOutput, vec);
    }

    return vec;
  }

  std::unique_ptr<layers::BERTEmbedding> embedding_;
  std::vector<BERTLayer> encoders_;
  std::unique_ptr<layers::BertPooler> pooler_;

  DLDeviceType device_type_;
};

BertModel::BertModel(const std::string &filename, DLDeviceType device_type,
                     size_t n_layers, int64_t n_heads)
    : m_(new Impl(filename, device_type, n_layers, n_heads)) {}
std::vector<float> BertModel::operator()(
    const std::vector<std::vector<int64_t>> &inputs,
    const std::vector<std::vector<int64_t>> &poistion_ids,
    const std::vector<std::vector<int64_t>> &segment_ids, PoolType pooling,

    bool use_pooler) const {
  return m_->operator()(inputs, poistion_ids, segment_ids, pooling, use_pooler);
}
BertModel::~BertModel() = default;
}  // namespace loaders
}  // namespace turbo_transformers
