import unittest

import onnxruntime
import torch
from torch import nn
import torch.jit
import torch.onnx
import torch.utils.dlpack as dlpack
from onnxruntime.backend import backend
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertLayer
import contexttimer
import fast_transformers
import caffe2.python.onnx.backend as caffe2_backend
import onnx


# dlpack to fast_transformer inline Tensor
def _(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))

class FastBertLayer(nn.Module):
    def __init__(self, config, bert_layer_params, isSelf: bool):
        super(FastBertLayer, self).__init__()
        # TODO: currently not support decoder



        # Optimization for self-attention
        if(isSelf):
            qkv_weight = torch.cat((bert_layer_params['attention.self.query.weight'], bert_layer_params['attention.self.key.weight']), 0)
            qkv_weight   = torch.cat((qkv_weight, bert_layer_params['attention.self.value.weight']), 0)

            qkv_bias = torch.cat((bert_layer_params['attention.self.query.bias'], bert_layer_params['attention.self.key.bias']), 0)
            qkv_bias   = torch.cat((qkv_bias, bert_layer_params['attention.self.value.bias']), 0)

            self.attention = fast_transformers.BertSelfAttention(_(qkv_weight),
                                            _(qkv_bias),
                                            _((bert_layer_params['attention.output.dense.weight'])),
                                            _(bert_layer_params['attention.output.dense.bias']),
                                            _(bert_layer_params['attention.output.LayerNorm.weight']),
                                            _(bert_layer_params['attention.output.LayerNorm.bias']),
                                            config.num_attention_heads)
        else:
            self.attention = fast_transformers.BertAttention(_((bert_layer_params['attention.self.query.weight'])),
                                        _(bert_layer_params['attention.self.query.bias']),
                                        _((bert_layer_params['attention.self.key.weight'])),
                                        _(bert_layer_params['attention.self.key.bias']),
                                        _((bert_layer_params['attention.self.value.weight'])),
                                        _(bert_layer_params['attention.self.value.bias']),
                                        _((bert_layer_params['attention.output.dense.weight'])),
                                        _(bert_layer_params['attention.output.dense.bias']),
                                        _(bert_layer_params['attention.output.LayerNorm.weight']),
                                        _(bert_layer_params['attention.output.LayerNorm.bias']),
                                        config.num_attention_heads)

        self.intermediate = fast_transformers.BertIntermediate(_(bert_layer_params['intermediate.dense.weight']),
                                                               _(bert_layer_params['intermediate.dense.bias'])
                                                               )

        self.output = fast_transformers.BertOutput(_((bert_layer_params["output.dense.weight"])),  # trans here Important
                                                        _(bert_layer_params["output.dense.bias"]),  # bertout_params["dense.bias"],
                                                        _(bert_layer_params["output.LayerNorm.weight"]),
                                                        _(bert_layer_params["output.LayerNorm.bias"])
                                                        )

    # do not output weight
    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(_(hidden_states), _(attention_mask), _(head_mask))
        attention_output = self_attention_outputs #[0] TODO
        #outputs = self_attention_outputs[1:]  # TODO add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        #outputs = (layer_output,) + outputs # TODO
        return layer_output


class TestBertLayer(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_grad_enabled(False)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.cfg = BertConfig(vocab_size_or_config_json_file=self.tokenizer.vocab_size,
                              attention_probs_dropout_prob=0.0,
                              hidden_dropout_prob=0.0)

        self.torch_bert_layer = BertLayer(self.cfg)
        self.torch_bert_layer.eval()

        self.batch_size = 20
        self.seq_length = 40
        self.hidden_size = self.cfg.hidden_size
        self.input_tensor = torch.rand(size=(self.batch_size, self.seq_length, self.hidden_size), dtype=torch.float32)

        self.attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)
        self.attention_mask = self.attention_mask[:, None, None, :]
        self.attention_mask = (1.0 - self.attention_mask) * -10000.0
        self.head_mask = torch.ones((self.batch_size, self.cfg.num_attention_heads, self.seq_length, self.seq_length), dtype = torch.long)

        # build fast attention
        # I did not set encoder_hidden_states and encoder_attention_mask.
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        bert_layer_params = {k: v for k, v in
                            self.torch_bert_layer.named_parameters()}

        self.ft_bert_layer = FastBertLayer(self.cfg, bert_layer_params, True)

    def test_bert_layer(self):
        self.num_iter = 100

        torch_bert_layer_result = self.torch_bert_layer(self.input_tensor, self.attention_mask, self.head_mask)
        with contexttimer.Timer() as t:
            for it in range(self.num_iter):
                torch_bert_layer_result = self.torch_bert_layer(self.input_tensor, self.attention_mask, self.head_mask)

        print(f"BertLayer Torch QPS, {self.num_iter / t.elapsed}, ", f"Time Cost, {t.elapsed / self.num_iter}")

        fast_transformers.enable_gperf("./profile.perf")
        ft_bert_layer_result = self.ft_bert_layer(self.input_tensor, self.attention_mask, self.head_mask)
        with contexttimer.Timer() as t:
            for it in range(self.num_iter):
                ft_bert_layer_result = self.ft_bert_layer(self.input_tensor, self.attention_mask, self.head_mask)

        fast_transformers.disable_gperf()

        print(f"fast_transformer BertLayer Torch QPS, {self.num_iter / t.elapsed}, ", f"Time Cost, {t.elapsed / self.num_iter}")

        # # fast_transformer
        # ft_attention_result = dlpack.from_dlpack(
        #     self.ft_attention(_(self.input_tensor), _(self.attention_mask), _(self.head_mask)).to_dlpack());
        # with contexttimer.Timer() as t:
        #     for it in range(self.num_iter):
        #         ft_attention_result = dlpack.from_dlpack(
        #             self.ft_attention(_(self.input_tensor), _(self.attention_mask), _(self.head_mask)).to_dlpack());

        # print(f"BertAttention FastTransform QPS {self.num_iter / t.elapsed}")
        ft_bert_layer_result = dlpack.from_dlpack(ft_bert_layer_result.to_dlpack())
        #print("diff is: ", torch.max(torch.abs( torch_bert_layer_result[0] - ft_bert_layer_result) ),
        #        " avg is : ", torch.mean(torch.abs( torch_bert_layer_result[0] - ft_bert_layer_result) ))
        self.assertTrue( torch.max( torch.abs( torch_bert_layer_result[0] - ft_bert_layer_result) ) < 1e-3 )


if __name__ == '__main__':
    unittest.main()
