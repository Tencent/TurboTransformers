// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#include "bert_model.h"

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

using namespace turbo_transformers::loaders;

static std::unique_ptr<layers::BERTEmbedding> LoadEmbedding(NPZMapView npz,
                                                            DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BERTEmbedding>(new layers::BERTEmbedding(
      params["word_embeddings.weight"], params["position_embeddings.weight"],
      params["token_type_embeddings.weight"], params["LayerNorm.weight"],
      params["LayerNorm.bias"]));
}

static std::unique_ptr<layers::BertPooler> LoadPooler(NPZMapView npz,
                                                      DLDeviceType dev) {
  NPZLoader params(std::move(npz), dev);

  return std::unique_ptr<layers::BertPooler>(
      new layers::BertPooler(params["dense.weight"], params["dense.bias"]));
}

struct BERTLayer {
  explicit BERTLayer(NPZLoader params, int64_t n_heads) {
    // define layer network here
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

    // HERE define your network model
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

  // preprocess helper function
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

  // do inference
  std::vector<float> operator()(
      const std::vector<std::vector<int64_t>> &inputs,
      const std::vector<std::vector<int64_t>> &poistion_ids,
      const std::vector<std::vector<int64_t>> &segment_ids, PoolType pooling,
      bool use_pooler) {
    core::Tensor inputs_tensor{nullptr};
    core::Tensor masks_tensor{nullptr};

    core::Tensor gpuInputs_tensor{nullptr};
    core::Tensor gpuMasks_tensor{nullptr};

    int64_t max_seq_len =
        std::accumulate(inputs.begin(), inputs.end(), 0,
                        [](size_t len, const std::vector<int64_t> &input_ids) {
                          return std::max(len, input_ids.size());
                        });
    int64_t batch_size = inputs.size();
    auto *iptr = inputs_tensor.Reshape<int64_t>(
        {batch_size, max_seq_len}, DLDeviceType::kDLCPU, 0,
        "PrepareBertMasks/seqids/Reshape");
    auto *mptr = masks_tensor.Reshape<int64_t>(
        {batch_size, max_seq_len}, DLDeviceType::kDLCPU, 0,
        "PrepareBertMasks/attmask/Reshape");

    for (size_t i = 0; i < inputs.size();
         ++i, iptr += max_seq_len, mptr += max_seq_len) {
      auto &input = inputs[i];
      // TODO(jiaruifang) Bert_Attention use mask value as 1 to indicate a valid
      // position.
      std::copy(input.begin(), input.end(), iptr);
      std::fill(mptr, mptr + input.size(), 1);
      if (input.size() != static_cast<size_t>(max_seq_len)) {
        std::fill(iptr + input.size(), iptr + max_seq_len, 0);
        std::fill(mptr + input.size(), mptr + max_seq_len, 0);
      }
    }
    if (device_type_ == DLDeviceType::kDLGPU) {
      gpuInputs_tensor.Reshape<int64_t>({batch_size, max_seq_len},
                                        DLDeviceType::kDLGPU, 0,
                                        "PrepareBertMasks/seqids/Reshape");
      gpuMasks_tensor.Reshape<int64_t>({batch_size, max_seq_len},
                                       DLDeviceType::kDLGPU, 0,
                                       "PrepareBertMasks/attmask/Reshape");
      core::Copy(inputs_tensor.data<int64_t>(), inputs_tensor.numel(),
                 DLDeviceType::kDLCPU, gpuInputs_tensor);
      core::Copy(masks_tensor.data<int64_t>(), masks_tensor.numel(),
                 DLDeviceType::kDLCPU, gpuMasks_tensor);
    }
    auto &inputIds =
        device_type_ == DLDeviceType::kDLCPU ? inputs_tensor : gpuInputs_tensor;

    core::Tensor seqType(nullptr);
    core::Tensor positionIds(nullptr);
    core::Tensor extendedAttentionMask(nullptr);
    if (poistion_ids.size() != 0) {
      TT_ENFORCE_EQ(
          poistion_ids.size(), static_cast<size_t>(batch_size),
          "Position ids should have the same batch size as input ids");
      PadTensor(poistion_ids, batch_size, max_seq_len, static_cast<int64_t>(0),
                device_type_, &positionIds);
    }
    if (segment_ids.size() != 0) {
      TT_ENFORCE_EQ(segment_ids.size(), static_cast<size_t>(batch_size),
                    "Segment ids should have the same batch size as input ids");
      PadTensor(segment_ids, batch_size, max_seq_len, static_cast<int64_t>(0),
                device_type_, &seqType);
    }

    layers::PrepareBertMasks()(
        inputIds,
        device_type_ == DLDeviceType::kDLCPU ? &masks_tensor : &gpuMasks_tensor,
        &seqType, &positionIds, &extendedAttentionMask);

    // start inference the BERT
    core::Tensor hidden(nullptr);
    (*embedding_)(inputIds, positionIds, seqType, &hidden);
    core::Tensor attOut(nullptr);
    core::Tensor intermediateOut(nullptr);
    for (auto &layer : encoders_) {
      layer(hidden, extendedAttentionMask, &attOut, &intermediateOut, &hidden);
    }

    std::vector<float> vec;
    if (use_pooler) {
      core::Tensor output(nullptr);
      core::Tensor poolingOutput(nullptr);
      layers::SequencePool(static_cast<layers::types::PoolType>(pooling))(
          hidden, &poolingOutput);
      (*pooler_)(poolingOutput, &output);
      vec.resize(output.numel());
      core::Copy(output, vec);
    } else {
      vec.resize(hidden.numel());
      core::Copy(hidden, vec);
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
