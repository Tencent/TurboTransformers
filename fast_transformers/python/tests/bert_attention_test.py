import unittest

import contexttimer
import torch
import torch.jit
import torch.onnx
from transformers.modeling_bert import BertConfig, BertAttention
import fast_transformers
from transformers import BertTokenizer
import onnxruntime.backend
import os


def create_test(batch_size, seq_length):
    class TestBertAttention(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))

            # Get Torch attention
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size,
                attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0)
            self.torch_attention = BertAttention(self.cfg)
            self.torch_attention.eval()

            # Get FT Attention
            num_attention_heads = self.cfg.num_attention_heads
            self.ft_attention = fast_transformers.BertAttention.from_torch(
                self.torch_attention)

            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(batch_size, seq_length,
                                                 self.hidden_size),
                                           dtype=torch.float32)

            self.attention_mask = torch.ones((batch_size, seq_length),
                                             dtype=torch.long)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

            self.torch_attention(self.input_tensor, self.attention_mask)
            torch.onnx.export(self.torch_attention,
                              args=(self.input_tensor, self.attention_mask),
                              output_names=['out'],
                              f='bert-attn.onnx')

            # Get Torch JIT attention
            self.jit_attention = torch.jit.trace(
                self.torch_attention,
                example_inputs=(self.input_tensor, self.attention_mask))

            # Prepare the backend
            self.onnxruntime_attention = self.get_onnxruntime_modle(
                onnx_file="bert-attn.onnx")

        def test_bertattention(self):
            self.num_iter = 50
            # Torch
            model = lambda: self.torch_attention(self.input_tensor, self.
                                                 attention_mask)
            torch_attention_result = self.run_model("Torch", model)

            # Torch JIT
            model = lambda: self.jit_attention(self.input_tensor, self.
                                               attention_mask)
            with torch.jit.optimized_execution(True):
                model()
            torch_attention_result = self.run_model("Torch JIT", model)

            # ONNX(MKL-DNN)
            onnx_inputs = [
                t.numpy() for t in [self.input_tensor, self.attention_mask]
            ]

            self.onnxruntime_attention.run(inputs=onnx_inputs)

            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    self.onnxruntime_attention.run(inputs=onnx_inputs)
            print(
                f'BertAttention({batch_size}, {seq_length}) ONNX(MKL-DNN) QPS, {self.num_iter / t.elapsed}, Elapse {t.elapsed / self.num_iter}'
            )

            ft_self_attention_result = self.ft_attention(
                self.input_tensor, self.attention_mask)

            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    ft_self_attention_result = self.ft_attention(
                        self.input_tensor, self.attention_mask)
            print(
                f"BertAttention({batch_size}, {seq_length}) BertAttention QPS, {self.num_iter / t.elapsed}, Elapse {t.elapsed / self.num_iter}"
            )
            # print("max diff: ", torch.max(torch.abs(torch_attention_result[0] - ft_self_attention_result)))
            self.assertTrue(
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              ft_self_attention_result)) < 1e-4)

        def run_model(self, model_name, model, num_iter=50):
            # warmup
            model()
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_attention_result = model()
            print("BertAttention({}, {}) {} QPS, {}, Elapse{}".format(
                batch_size, seq_length, model_name, num_iter / t.elapsed,
                t.elapsed / num_iter))
            return torch_attention_result

        def get_onnxruntime_modle(self, onnx_file):
            if not onnxruntime.backend.supports_device("MKL-DNN"):
                return onnxruntime.backend.prepare(onnx_file, device="CPU")
            else:
                return onnxruntime.backend.prepare(onnx_file, device="MKL-DNN")

    globals()[f"TestBertAtt{batch_size}_{seq_length:3}"] = TestBertAttention


for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
