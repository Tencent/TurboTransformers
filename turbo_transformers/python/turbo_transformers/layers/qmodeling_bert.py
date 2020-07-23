import turbo_transformers.turbo_transformers_cxx as cxx
import torch
from .return_type import convert_returns_as_type, ReturnType
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor
from .modeling_bert import BertAttention

__all__ = [
    'QBertIntermediate', 'QBertOutput', 'QBertLayer'
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
