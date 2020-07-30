import turbo_transformers.turbo_transformers_cxx as cxx
import torch
from .return_type import convert_returns_as_type, ReturnType
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor
from .modeling_bert import BertAttention, BertEmbeddings, BertEncoder, BertPooler, SequencePool, PoolingType, PoolingMap

__all__ = [
    'QBertIntermediate', 'QBertOutput', 'QBertLayer', 'QBertEncoder', 'QBertModel'
]


class QBertIntermediate:
    def __init__(self, intermediate):
        # assert intermediate.intermediate_act_fn is gelu
        self.bias_act = cxx.FusedAddBiasGELU(convert2tt_tensor(intermediate.dense.bias))
        self.qlinear = torch.quantization.quantize_dynamic(intermediate).dense
        self.qlinear.set_weight_bias(self.qlinear.weight(), None)
    def __call__(self, input_tensor):
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = convert_returns_as_type(input_tensor, ReturnType.TORCH)
        output = convert2tt_tensor(self.qlinear(input_tensor))
        self.bias_act(output)
        return convert_returns_as_type(output, ReturnType.TORCH)
    @staticmethod
    def from_torch(intermediate):
        return QBertIntermediate(intermediate)

class QBertOutput:
    def __init__(self, bert_output):
        self.bias_layernorm = cxx.FusedAddBiasLayerNorm(
            convert2tt_tensor(bert_output.dense.bias),
            convert2tt_tensor(bert_output.LayerNorm.weight),
            convert2tt_tensor(bert_output.LayerNorm.bias))
        self.qlinear = torch.quantization.quantize_dynamic(bert_output).dense
        self.qlinear.set_weight_bias(self.qlinear.weight(), None)
    def __call__(self, intermediate_output, attention_output):
        if not isinstance(intermediate_output, torch.Tensor):
            intermediate_output = convert_returns_as_type(intermediate_output, ReturnType.TORCH)
        output = convert2tt_tensor(self.qlinear(intermediate_output))
        self.bias_layernorm(convert2tt_tensor(attention_output), output)
        return convert_returns_as_type(output, ReturnType.TORCH)
    @staticmethod
    def from_torch(bert_output):
        return QBertOutput(bert_output)

class QBertLayer:
    def __init__(self, bert_layer):
        self.attention = BertAttention.from_torch(bert_layer.attention)
        self.intermediate = QBertIntermediate.from_torch(bert_layer.intermediate)
        self.output = QBertOutput.from_torch(bert_layer.output)
    def __call__(self,
                 hidden_states,
                 attention_mask=None,
                 head_mask=None,
                 output_attentions=False):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            return_type=ReturnType.TORCH)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + outputs
        return outputs
    @staticmethod
    def from_torch(bert_layer):
        return QBertLayer(bert_layer)

class QBertEncoder:
    def __init__(self, layers):
        self.layers = layers
    def __call__(self,
                 hidden_states,
                 attention_mask = None,
                 head_mask = None,
                 output_attentions = False,
                 output_hidden_states = False):
        all_hidden_states = ()
        all_attentions = ()
        for l in self.layers:
            layer_outputs = l(hidden_states=hidden_states,
                              attention_mask=attention_mask,
                              output_attentions=output_attentions)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    convert_returns_as_type(hidden_states, ReturnType.TORCH), )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )
        outputs = (hidden_states, )
        if output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs
    @staticmethod
    def from_torch(encoder):
        layers = [QBertLayer.from_torch(bert_layer) for bert_layer in encoder.layer]
        return QBertEncoder(layers)

class QBertModel:
    def __init__(self, model):
        self.embeddings = BertEmbeddings.from_torch(model.embeddings)
        self.encoder = QBertEncoder.from_torch(model.encoder)
        self.pooler = BertPooler.from_torch(model.pooler)
        self.prepare = cxx.PrepareBertMasks()
    def __call__(self, inputs,
                 attention_masks = None,
                 token_type_ids = None,
                 position_ids = None,
                 head_mask = None,
                 inputs_embeds = None,
                 output_attentions = None,
                 output_hidden_states = None,
                 pooling_type = PoolingType.FIRST,
                 pooler_output = None):
        attention_masks = try_convert(create_empty_if_none(attention_masks))
        token_type_ids = try_convert(create_empty_if_none(token_type_ids))
        position_ids = try_convert(create_empty_if_none(position_ids))
        inputs = try_convert(inputs)
        extended_attention_masks = cxx.Tensor.create_empty()
        self.prepare(inputs, attention_masks, token_type_ids, position_ids, extended_attention_masks)
        hidden_cache = self.embeddings(
            inputs,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            return_type=ReturnType.TORCH)
        encoder_outputs = self.encoder(
            hidden_states=hidden_cache,
            attention_mask=extended_attention_masks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        sequence_output = encoder_outputs[0]
        self.seq_pool = SequencePool(PoolingMap[pooling_type])
        sequence_pool_output = self.seq_pool(
            input_tensor=sequence_output,
            return_type=ReturnType.TORCH)
        pooler_output = self.pooler(sequence_pool_output, ReturnType.TORCH,
                                    pooler_output)
        return (sequence_output, pooler_output, ) + encoder_outputs[1:]
    @staticmethod
    def from_torch(model):
        return QBertModel(model)
