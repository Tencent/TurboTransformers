# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
from typing import Union, Optional, Sequence
import torch
from .return_type import convert_returns_as_type, ReturnType

from .utils import try_convert, convert2tt_tensor, create_empty_if_none, AnyTensor

from onmt.modules.multi_headed_attn import MultiHeadedAttention as OnmtMultiHeadedAttention
from transformers.modeling_bert import BertAttention as TorchBertAttention

from onmt.modules.position_ffn import PositionwiseFeedForward as OnmtPositionwiseFeedForward
from onmt.decoders.transformer import TransformerDecoderLayer as OnmtTransformerDecoderLayer
from onmt.decoders.transformer import TransformerDecoder as OnmtTransformerDecoder
from onmt.modules import Embeddings as TorchBertEmbeddings

from torch.nn import LayerNorm as TorchLayerNorm
from onmt.utils.misc import sequence_mask

import enum
import numpy as np

__all__ = [
    'MultiHeadedAttention', 'PositionwiseFeedForward',
    'TransformerDecoderLayer', 'TransformerDecoder'
]


class MultiHeadedAttention(cxx.MultiHeadedAttention):
    def __call__(self,
                 key_tensor: AnyTensor,
                 value_tensor: AnyTensor,
                 query_tensor: AnyTensor,
                 mask: Optional[AnyTensor] = None,
                 layer_cache: Optional[dict] = None,
                 attn_type: str = None,
                 pre_layernorm: bool = False,
                 post_layernorm: bool = False,
                 post_add_input: bool = False,
                 is_trans_weight: bool = False,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None,
                 attn: Optional[cxx.Tensor] = None):
        """ Implement a MultiHeadedAttention of OpenNMT-py
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/multi_headed_attn.py

        Attention: Now layer_cache only contains Nones
        For self-dot Attention elements in dict `layer_cache` are Nones.
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py#L339
        """
        key_tensor = try_convert(key_tensor)
        value_tensor = try_convert(value_tensor)
        query_tensor = try_convert(query_tensor)

        mask = try_convert(create_empty_if_none(mask))

        output = create_empty_if_none(output)
        attn = create_empty_if_none(attn)
        layer_cache_tmp = {}
        if layer_cache is not None:
            for k, v in layer_cache.items():
                if v is not None:
                    layer_cache_tmp[k] = try_convert(v)
                else:
                    layer_cache_tmp[k] = create_empty_if_none(v)

        super(MultiHeadedAttention,
              self).__call__(key_tensor, value_tensor, query_tensor, mask,
                             attn_type, output, attn, layer_cache_tmp,
                             pre_layernorm, post_layernorm, post_add_input,
                             is_trans_weight)

        if layer_cache is not None:
            for k, v in layer_cache_tmp.items():
                if "memory" in k and "context" in attn_type or "self" in k and "self" in attn_type:
                    layer_cache[k] = convert_returns_as_type(
                        v, ReturnType.TORCH)

        return convert_returns_as_type(output,
                                       return_type), convert_returns_as_type(
                                           attn, return_type)

    @staticmethod
    def pack_parameter(multi_headed_attn: OnmtMultiHeadedAttention,
                       is_trans_weight: Optional[bool] = False):
        # linear_keys.weight
        # linear_keys.bias
        # linear_values.weight
        # linear_values.bias
        # linear_query.weight
        # linear_query.bias
        # final_linear.weight
        # final_linear.bias
        attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
        if multi_headed_attn.max_relative_positions != 0:
            raise "multi_headed_attn's max_relative_positions should be 0!"

        # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
        if is_trans_weight:
            qkv_weight = torch.cat((attn_params['linear_query.weight'],
                                    attn_params['linear_keys.weight'],
                                    attn_params['linear_values.weight']), 0)
            k_w = convert2tt_tensor(attn_params['linear_keys.weight'])
            v_w = convert2tt_tensor(attn_params['linear_values.weight'])
            q_w = convert2tt_tensor(attn_params['linear_query.weight'])
            f_w = convert2tt_tensor(attn_params['final_linear.weight'])
        else:
            qkv_weight = torch.clone(
                torch.t(
                    torch.cat((attn_params['linear_query.weight'],
                               attn_params['linear_keys.weight'],
                               attn_params['linear_values.weight']),
                              0).contiguous()).contiguous())
            k_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['linear_keys.weight']).contiguous()))
            v_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['linear_values.weight']).contiguous()))
            q_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['linear_query.weight']).contiguous()))
            f_w = convert2tt_tensor(
                torch.clone(
                    torch.t(attn_params['final_linear.weight']).contiguous()))

        qkv_bias = torch.cat(
            (attn_params['linear_query.bias'], attn_params['linear_keys.bias'],
             attn_params['linear_values.bias']), 0)
        return (k_w, convert2tt_tensor(attn_params['linear_keys.bias']), v_w,
                convert2tt_tensor(attn_params['linear_values.bias']), q_w,
                convert2tt_tensor(attn_params['linear_query.bias']), f_w,
                convert2tt_tensor(attn_params['final_linear.bias']),
                convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias))

    @staticmethod
    def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention,
                  is_trans_weight: bool = False):
        attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
        if multi_headed_attn.max_relative_positions != 0:
            raise "multi_headed_attn's max_relative_positions should be 0!"

        with torch.no_grad():
            att = MultiHeadedAttention(
                *(MultiHeadedAttention.pack_parameter(attn_params,
                                                      is_trans_weight)),
                multi_headed_attn.head_count)
            return att

    @staticmethod
    def from_onmt(multi_headed_attn: OnmtMultiHeadedAttention,
                  layer_norm: TorchLayerNorm,
                  is_trans_weight: bool = False):
        ln_params = {k: v for k, v in layer_norm.named_parameters()}
        attn_params = {k: v for k, v in multi_headed_attn.named_parameters()}
        with torch.no_grad():
            att = MultiHeadedAttention(
                *(MultiHeadedAttention.pack_parameter(multi_headed_attn,
                                                      is_trans_weight)),
                convert2tt_tensor(ln_params['weight']),
                convert2tt_tensor(ln_params['bias']),
                multi_headed_attn.head_count)
            return att

    @staticmethod
    def from_torch(attention: TorchBertAttention,
                   layer_norm: Optional[TorchLayerNorm] = None,
                   is_trans_weight: bool = False):
        """
        load an attn model from huggingface bert attention model.
        """
        ln_params = {}
        if layer_norm is not None:
            ln_params = {k: v for k, v in layer_norm.named_parameters()}
        params = {k: v for k, v in attention.named_parameters()}
        with torch.no_grad():
            if is_trans_weight:
                # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
                qkv_weight = torch.cat(
                    (params['self.query.weight'], params['self.key.weight'],
                     params['self.value.weight']), 0)
                output_weight = params['output.dense.weight']
                k_w = params['self.key.weight']
                v_w = params['self.value.weight']
                q_w = params['self.query.weight']
            else:
                # merge self.query.weight, self.query.weight and self.query.weight together as qkv.weight
                qkv_weight = torch.clone(
                    torch.t(
                        torch.cat((params['self.query.weight'],
                                   params['self.key.weight'],
                                   params['self.value.weight']),
                                  0).contiguous()).contiguous())
                output_weight = torch.clone(
                    torch.t(params['output.dense.weight']).contiguous())
                k_w = torch.clone(
                    torch.t(params['self.key.weight']).contiguous())
                v_w = torch.clone(
                    torch.t(params['self.value.weight']).contiguous())
                q_w = torch.clone(
                    torch.t(params['self.query.weight']).contiguous())

            qkv_bias = torch.cat(
                (params['self.query.bias'], params['self.key.bias'],
                 params['self.value.bias']), 0)

            if layer_norm is not None:
                att = MultiHeadedAttention(
                    convert2tt_tensor(k_w),
                    convert2tt_tensor(params['self.key.bias']),
                    convert2tt_tensor(v_w),
                    convert2tt_tensor(params['self.value.bias']),
                    convert2tt_tensor(q_w),
                    convert2tt_tensor(params['self.query.bias']),
                    convert2tt_tensor(output_weight),
                    convert2tt_tensor(params['output.dense.bias']),
                    convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                    convert2tt_tensor(params['output.LayerNorm.weight']),
                    convert2tt_tensor(params['output.LayerNorm.bias']),
                    convert2tt_tensor(ln_params['weight']),
                    convert2tt_tensor(ln_params['bias']),
                    attention.self.num_attention_heads)
            else:
                att = MultiHeadedAttention(
                    convert2tt_tensor(k_w),
                    convert2tt_tensor(params['self.key.bias']),
                    convert2tt_tensor(v_w),
                    convert2tt_tensor(params['self.value.bias']),
                    convert2tt_tensor(q_w),
                    convert2tt_tensor(params['self.query.bias']),
                    convert2tt_tensor(output_weight),
                    convert2tt_tensor(params['output.dense.bias']),
                    convert2tt_tensor(qkv_weight), convert2tt_tensor(qkv_bias),
                    convert2tt_tensor(params['output.LayerNorm.weight']),
                    convert2tt_tensor(params['output.LayerNorm.bias']),
                    attention.self.num_attention_heads)
            return att

    @staticmethod
    def from_npz(file_name: str, layer_num: int, num_attention_heads: int):
        f = np.load(file_name)
        return BertAttention(
            create_empty_if_none(None), create_empty_if_none(None),
            create_empty_if_none(None), create_empty_if_none(None),
            create_empty_if_none(None), create_empty_if_none(None),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.weight']),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.dense.bias']),
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.weight']),
            try_convert(f[f'encoder.layer.{layer_num}.attention.qkv.bias']),
            try_convert(f[
                f'encoder.layer.{layer_num}.attention.output.LayerNorm.weight']
                        ),
            try_convert(
                f[f'encoder.layer.{layer_num}.attention.output.LayerNorm.bias']
            ), num_attention_heads)


class PositionwiseFeedForward(cxx.PositionwiseFeedForward):
    def __call__(
            self,
            input_tensor: AnyTensor,
            return_type: Optional[ReturnType] = None,
            is_trans_weight: Optional[bool] = True,  #Intel 61xx True is faster
            output: Optional[cxx.Tensor] = None):
        input_tensor = try_convert(input_tensor)
        output = create_empty_if_none(output)
        super(PositionwiseFeedForward, self).__call__(input_tensor, output,
                                                      is_trans_weight)
        return convert_returns_as_type(output, return_type)

    @staticmethod
    def from_onmt(position_wise_ffn: OnmtPositionwiseFeedForward,
                  is_trans_weight: Optional[bool] = True):
        params = {k: v for k, v in position_wise_ffn.named_parameters()}
        # w_1.weight
        # w_1.bias
        # w_2.weight
        # w_2.bias
        # layer_norm.weight
        # layer_norm.bias

        # Note that torch's weights of linear layer is transposed
        if is_trans_weight:
            w_1 = convert2tt_tensor(params['w_1.weight'])
            w_2 = convert2tt_tensor(params['w_2.weight'])
        else:
            w_1 = convert2tt_tensor(
                torch.clone(torch.t(params['w_1.weight']).contiguous()))
            w_2 = convert2tt_tensor(
                torch.clone(torch.t(params['w_2.weight']).contiguous()))

        with torch.no_grad():
            ffn = PositionwiseFeedForward(
                w_1, convert2tt_tensor(params['w_1.bias']), w_2,
                convert2tt_tensor(params['w_2.bias']),
                convert2tt_tensor(params['layer_norm.weight']),
                convert2tt_tensor(params['layer_norm.bias']))
            return ffn


# The onnxruntime dose not support (FloatTensor, FloatTensor, FloatTensor or None)
# The first two FloatTensors are kept.
class ModifiedOnmtTransformerDecoderLayer(torch.nn.Module):
    def __init__(self, model):
        super(ModifiedOnmtTransformerDecoderLayer, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)[:-1]


class TransformerDecoderLayer:
    def __init__(self,
                 self_attn: MultiHeadedAttention,
                 context_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward,
                 model=None,
                 backend='turbo'):
        """ Implement class TransformerDecoderLayer(nn.Module):
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py
        self_attn_type of MultiHeadedAttention should always scaled-dot
        """
        if backend == 'onnxrt':
            self.backend = 'onnxrt'
            d_model = model.layer_norm_1.normalized_shape[0]
            # trick
            model = ModifiedOnmtTransformerDecoderLayer(model)
            dummy_input = {
                'input_tensor': torch.rand(1, 10, d_model,
                                           dtype=torch.float32),
                'memory_bank': torch.rand(1, 10, d_model, dtype=torch.float32),
                'src_pad_mask': torch.zeros(1, 1, 10, dtype=torch.bool),
                'dec_mask': torch.zeros(1, 1, 10, dtype=torch.bool)
            }
            symbolic_names = {0: 'batch_size', 1: 'max_len'}
            symbolic_names_2 = {0: 'batch_size', 2: 'max_len'}
            self.onnx_model_path = "/tmp/temp_turbo_onnx.model"
            with open(self.onnx_model_path, 'wb') as f:
                torch.onnx.export(
                    model,
                    (dummy_input['input_tensor'], dummy_input['memory_bank'],
                     dummy_input['src_pad_mask'], dummy_input['dec_mask']),
                    f,
                    input_names=[
                        'input_tensor', 'memory_bank', 'src_pad_mask',
                        'dec_mask'
                    ],
                    output_names=['output'],
                    opset_version=11,
                    dynamic_axes={
                        'input_tensor': symbolic_names,
                        'memory_bank': symbolic_names,
                        'src_pad_mask': symbolic_names_2,
                        'dec_mask': symbolic_names_2
                    })
            import onnxruntime
            sess_options = onnxruntime.SessionOptions()
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = onnxruntime.InferenceSession(
                self.onnx_model_path, sess_options)
        else:
            self.backend = 'turbo'
            self.self_attn = self_attn
            if not isinstance(self_attn, MultiHeadedAttention):
                raise "self_attn should be of type MultiHeadedAttention"
            self.context_attn = context_attn
            if not isinstance(context_attn, MultiHeadedAttention):
                raise "context_attn should be of type MultiHeadedAttention"
            self.feed_forward = feed_forward

    def quantize_dynamic(self):
        assert self.backend == 'onnxrt'
        from onnxruntime.quantization import quantize, QuantizationMode
        import onnx
        import onnxruntime
        import onnxruntime.backend
        opt_model = onnx.load(self.onnx_model_path)
        quantized_onnx_model = quantize(
            opt_model,
            quantization_mode=QuantizationMode.IntegerOps,
            symmetric_weight=True,
            force_fusions=True)
        quantized_model_path = "/tmp/temp_turbo_onnx_q.model"
        onnx.save(quantized_onnx_model, quantized_model_path)
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(quantized_model_path,
                                                    sess_options)

    def __call__(self,
                 input_tensor: torch.Tensor,
                 memory_bank: torch.Tensor,
                 src_pad_mask: torch.Tensor,
                 tgt_pad_mask: torch.Tensor,
                 layer_cache: Optional[dict] = None,
                 step: Optional[int] = None,
                 future: Optional[bool] = False,
                 with_align: Optional[bool] = False,
                 return_type: Optional[ReturnType] = None,
                 output: Optional[cxx.Tensor] = None):
        """ Implement _forward method of class TransformerDecoderLayer
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/decoders/transformer.py
        Because we now do not need context aligment, so we do not provide a forward method
        Args:
            input_tensor (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.
        Returns:
            (FloatTensor, FloatTensor):
            * output ``(batch_size, T, model_dim)``
            * top_attns ``(batch_size, T, src_len)``  or None
            * attn_align None
        """
        if self.backend == 'onnxrt':
            if step is None:
                if not future:
                    tgt_len = tgt_pad_mask.size(-1)
                    future_mask_numpy = np.ones(shape=(tgt_len, tgt_len),
                                                dtype=np.float32)
                    future_mask_numpy = np.triu(future_mask_numpy)
                    # TODO(jiaruifang) move to GPU if use cuda
                    future_mask = torch.tensor(
                        future_mask_numpy,
                        device=input_tensor.device).view(1, tgt_len, tgt_len)
                    dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
                else:  # only mask padding, result mask in (B, 1, T)
                    dec_mask = tgt_pad_mask
            else:
                # init a dummy dec_mask
                dec_mask = torch.zeros(input_tensor.size(0),
                                       1,
                                       input_tensor.size(1),
                                       dtype=torch.float32,
                                       device=input_tensor.device).bool()

            ort_inputs = {
                'input_tensor': input_tensor.cpu().numpy(),
                'memory_bank': memory_bank.cpu().numpy(),
                'src_pad_mask': src_pad_mask.cpu().numpy(),
                'dec_mask': dec_mask.cpu().numpy()
            }
            return self.session.run(None, ort_inputs)

        # dec_mask = None which is no mask
        dec_mask = None

        input_tensor = try_convert(input_tensor)
        memory_bank = try_convert(memory_bank)
        src_pad_mask = src_pad_mask.float() * -1e18
        src_pad_mask = try_convert(src_pad_mask)

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones([tgt_len, tgt_len],
                                         device=tgt_pad_mask.device,
                                         dtype=torch.float32)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                # try:
                #     future_mask = future_mask.bool()
                # except AttributeError:
                #     pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0).float()
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        if dec_mask is None:
            dec_mask = create_empty_if_none(dec_mask)
        else:
            dec_mask = dec_mask * -1e18
            dec_mask = try_convert(dec_mask)

        query, _ = self.self_attn(input_tensor,
                                  input_tensor,
                                  input_tensor,
                                  mask=dec_mask,
                                  layer_cache=layer_cache,
                                  attn_type="self",
                                  pre_layernorm=True,
                                  post_add_input=True,
                                  return_type=ReturnType.turbo_transformers)

        mid, attns = self.context_attn(
            memory_bank,
            memory_bank,
            query,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
            pre_layernorm=True,
            post_add_input=True,
            return_type=ReturnType.turbo_transformers)

        output = self.feed_forward(mid, return_type=return_type)
        return output, convert_returns_as_type(
            attns, return_type)[:, 0, :, :].contiguous(
            ), None  #attn_aligned mast be None

    @staticmethod
    def from_onmt(transformer_decoder_layer: OnmtTransformerDecoderLayer,
                  backend='onnxrt'):
        if backend == 'onnxrt':
            return TransformerDecoderLayer(None,
                                           None,
                                           None,
                                           model=transformer_decoder_layer,
                                           backend='onnxrt')
        params = {
            k: v
            for k, v in transformer_decoder_layer.named_parameters()
        }
        # for k, v in transformer_decoder_layer.named_parameters():
        #     print(k, v.size())

        # 12: self_attn.linear_keys.weight torch.Size([1024, 1024])
        # 12: self_attn.linear_keys.bias torch.Size([1024])
        # 12: self_attn.linear_values.weight torch.Size([1024, 1024])
        # 12: self_attn.linear_values.bias torch.Size([1024])
        # 12: self_attn.linear_query.weight torch.Size([1024, 1024])
        # 12: self_attn.linear_query.bias torch.Size([1024])
        # 12: self_attn.final_linear.weight torch.Size([1024, 1024])
        # 12: self_attn.final_linear.bias torch.Size([1024])
        # 12: context_attn.linear_keys.weight torch.Size([1024, 1024])
        # 12: context_attn.linear_keys.bias torch.Size([1024])
        # 12: context_attn.linear_values.weight torch.Size([1024, 1024])
        # 12: context_attn.linear_values.bias torch.Size([1024])
        # 12: context_attn.linear_query.weight torch.Size([1024, 1024])
        # 12: context_attn.linear_query.bias torch.Size([1024])
        # 12: context_attn.final_linear.weight torch.Size([1024, 1024])
        # 12: context_attn.final_linear.bias torch.Size([1024])
        # 12: feed_forward.w_1.weight torch.Size([1, 1024])
        # 12: feed_forward.w_1.bias torch.Size([1])
        # 12: feed_forward.w_2.weight torch.Size([1024, 1])
        # 12: feed_forward.w_2.bias torch.Size([1024])
        # 12: feed_forward.layer_norm.weight torch.Size([1024])
        # 12: feed_forward.layer_norm.bias torch.Size([1024])
        # 12: layer_norm_1.weight torch.Size([1024])
        # 12: layer_norm_1.bias torch.Size([1024])
        # 12: layer_norm_2.weight torch.Size([1024])
        # 12: layer_norm_2.bias torch.Size([1024])
        # 12: w_1.weight torch.Size([1, 1024])
        # 12: w_1.bias torch.Size([1])
        # 12: w_2.weight torch.Size([1024, 1])
        # 12: w_2.bias torch.Size([1024])
        # 12: layer_norm.weight torch.Size([1024])
        # 12: layer_norm.bias torch.Size([1024])

        self_attn = MultiHeadedAttention.from_onmt(
            transformer_decoder_layer.self_attn,
            transformer_decoder_layer.layer_norm_1)
        context_attn = MultiHeadedAttention.from_onmt(
            transformer_decoder_layer.context_attn,
            transformer_decoder_layer.layer_norm_2)
        feed_forward = PositionwiseFeedForward.from_onmt(
            transformer_decoder_layer.feed_forward)

        return TransformerDecoderLayer(self_attn,
                                       context_attn,
                                       feed_forward,
                                       backend='turbo')


class TransformerDecoder:
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations, TODO(jiaruifang) only support 0
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder, TODO(jiaruifang) only support False
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """
    def __init__(self,
                 embeddings: TorchBertEmbeddings,
                 transformer_layers: Sequence[TransformerDecoderLayer],
                 layer_norm: TorchLayerNorm,
                 copy_attn: Optional[bool] = False,
                 alignment_layer: Optional[int] = 0):
        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        self.transformer_layers = transformer_layers

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn  #bool
        self.layer_norm = layer_norm

        self.alignment_layer = alignment_layer

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def __call__(self,
                 tgt: torch.Tensor,
                 memory_bank: torch.Tensor,
                 step: Optional[int] = None,
                 **kwargs):
        """Decode, possibly stepwise."""
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        #Turbo add bool -> float
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        with_align = kwargs.pop('with_align', False)
        if with_align:
            raise "with_align must be False"
        attn_aligns = []

        # It's Turbo's show time!
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn, attn_align = layer(output,
                                             src_memory_bank,
                                             src_pad_mask,
                                             tgt_pad_mask,
                                             layer_cache=layer_cache,
                                             step=step,
                                             with_align=with_align)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        # Turbo finished.
        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        if with_align:
            attns["align"] = attn_aligns[self.alignment_layer]  # `(B, Q, K)`
            # attns["align"] = torch.stack(attn_aligns, 0).mean(0)  # All avg

        # TODO(OpenNMT-py) change the way attns is returned dict => list or tuple (onnx)

        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            if not isinstance(layer.self_attn, MultiHeadedAttention):
                raise "MultiHeadedAttention only not supported"
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache

    @staticmethod
    def from_onmt(model: OnmtTransformerDecoder,
                  device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        layers = [
            TransformerDecoderLayer.from_onmt(transformer_layer)
            for transformer_layer in model.transformer_layers
        ]
        return TransformerDecoder(model.embeddings, layers, model.layer_norm,
                                  model._copy, model.alignment_layer)
