import unittest
import torch
import torch.jit
import torch.onnx
from onnxruntime.backend import backend
import onnxruntime
import fast_transformers
from utils import convert2ft_tensor, load_bert_token

def get_ft_attention(attention_params, num_attention_heads):
    # merge self.query.weight, self.query.weight and self.query.weight togather as qkv.weight
    qkv_weight = torch.cat((attention_params['self.query.weight'],
                            attention_params['self.key.weight'],
                            attention_params['self.value.weight']), 0)
    qkv_bias = torch.cat((attention_params['self.query.bias'],
                          attention_params['self.key.bias'],
                          attention_params['self.value.bias']), 0)
    ft_attention = fast_transformers.BertAttention(
        convert2ft_tensor(qkv_weight),
        convert2ft_tensor(qkv_bias),
        convert2ft_tensor(attention_params['output.dense.weight']),
        convert2ft_tensor(attention_params['output.dense.bias']),
        convert2ft_tensor(attention_params['output.LayerNorm.weight']),
        convert2ft_tensor(attention_params['output.LayerNorm.bias']),
        num_attention_heads)
    return ft_attention

class BertAttentionBase(unittest.TestCase):
    def setUp(self):
        fast_transformers.set_stderr_verbose_level(1)
        torch.set_grad_enabled(False)
        self.tokenizer = load_bert_token()

    def get_onnxruntime_modle(self, onnx_file):
        if not onnxruntime.backend.supports_device("MKL-DNN"):
            return onnxruntime.backend.prepare(onnx_file, device="CPU")
        else:
            return onnxruntime.backend.prepare(onnx_file, device="MKL-DNN")
