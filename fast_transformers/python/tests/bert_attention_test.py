import unittest

import contexttimer
import onnxruntime
import torch
import torch.jit
import torch.onnx
import torch.utils.dlpack as dlpack
from onnxruntime.backend import backend
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertAttention

import fast_transformers


def _(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))


def create_test(batch_size, seq_length):
    class TestBertAttention(unittest.TestCase):
        def setUp(self) -> None:
            fast_transformers.set_stderr_verbose_level(1)
            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.cfg = BertConfig(vocab_size_or_config_json_file=self.tokenizer.vocab_size,
                                  attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)

            self.torch_attention = BertAttention(self.cfg)
            self.torch_attention.eval()

            attention_params = {k: v for k, v in
                                self.torch_attention.named_parameters()}

            # merge self.query.weight, self.query.weight and self.query.weight togather as qkv.weight
            qkv_weight = torch.cat((attention_params['self.query.weight'], attention_params['self.key.weight']), 0)
            qkv_weight = torch.cat((qkv_weight, attention_params['self.value.weight']), 0)
            qkv_bias = torch.cat((attention_params['self.query.bias'], attention_params['self.key.bias']), 0)
            qkv_bias = torch.cat((qkv_bias, attention_params['self.value.bias']), 0)

            self.ft_attention = fast_transformers.BertAttention(_((qkv_weight)),
                                                                         _(qkv_bias),
                                                                         _((attention_params['output.dense.weight'])),
                                                                         _(attention_params['output.dense.bias']),
                                                                         _(attention_params['output.LayerNorm.weight']),
                                                                         _(attention_params['output.LayerNorm.bias']),
                                                                         self.cfg.num_attention_heads)

            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(batch_size, seq_length, self.hidden_size),
                                           dtype=torch.float32)

            self.attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

            self.head_mask = torch.ones(
                (batch_size, self.cfg.num_attention_heads, seq_length, seq_length),
                dtype=torch.long)

            self.torch_attention(self.input_tensor, self.attention_mask, self.head_mask)

            self.jit_attention = torch.jit.trace(self.torch_attention,
                                                 example_inputs=(
                                                     self.input_tensor, self.attention_mask, self.head_mask))

            torch.onnx.export(self.torch_attention, args=(self.input_tensor, self.attention_mask, self.head_mask),
                              output_names=['out'], f='bert-attn.onnx')

            # Prepare the backend
            if not onnxruntime.backend.supports_device("MKL-DNN"):
                self.onnxruntime_attention = onnxruntime.backend.prepare("bert-attn.onnx", device="CPU")
                # raise RuntimeError("Please recompile onnx-runtime to support MKL-DNN")
            else:
                self.onnxruntime_attention = onnxruntime.backend.prepare("bert-attn.onnx", device="MKL-DNN")

        def test_bertattention(self):
            self.num_iter = 50

            torch_attention_result = self.torch_attention(self.input_tensor, self.attention_mask, self.head_mask)
            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    torch_attention_result = self.torch_attention(self.input_tensor, self.attention_mask,
                                                                  self.head_mask)

            print(
                f"BertAttention({batch_size}, {seq_length}) Torch QPS, {self.num_iter / t.elapsed}, Elapse {t.elapsed / self.num_iter}")

            # JIT
            with torch.jit.optimized_execution(True):
                self.jit_attention(self.input_tensor, self.attention_mask, self.head_mask)
            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    torch_attention_result = self.jit_attention(self.input_tensor, self.attention_mask, self.head_mask)

            print(
                f"BertAttention({batch_size}, {seq_length}) JIT QPS, {self.num_iter / t.elapsed}, Elapse {t.elapsed / self.num_iter}")

            onnx_inputs = [t.numpy() for t in [self.input_tensor, self.attention_mask, self.head_mask]]

            self.onnxruntime_attention.run(inputs=onnx_inputs)

            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    self.onnxruntime_attention.run(inputs=onnx_inputs)
            print(
                f'BertAttention({batch_size}, {seq_length}) ONNX(MKL-DNN) QPS, {self.num_iter / t.elapsed}, Elapse {t.elapsed / self.num_iter}')

            ft_self_attention_result = dlpack.from_dlpack(
                self.ft_attention(_(self.input_tensor), _(self.attention_mask), _(self.head_mask)).to_dlpack());
            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    ft_self_attention_result = dlpack.from_dlpack(
                        self.ft_attention(_(self.input_tensor), _(self.attention_mask),
                                               _(self.head_mask)).to_dlpack());
            print(
                f"BertAttention({batch_size}, {seq_length}) BertAttention QPS, {self.num_iter / t.elapsed}, Elapse {t.elapsed / self.num_iter}")
            # print("max diff: ", torch.max(torch.abs(torch_attention_result[0] - ft_self_attention_result)))
            self.assertTrue(torch.max(torch.abs(torch_attention_result[0] - ft_self_attention_result)) < 1e-4)

    globals()[f"TestBertAtt{batch_size}_{seq_length:3}"] = TestBertAttention


for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
