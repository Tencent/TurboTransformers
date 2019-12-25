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
            self.gpu_device = torch.device('cuda:0')
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
            self.torch_attention.to(self.gpu_device)

            # Get FT Attention
            num_attention_heads = self.cfg.num_attention_heads
            self.ft_attention = fast_transformers.BertAttention.from_torch(
                self.torch_attention)

            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(batch_size, seq_length,
                                                 self.hidden_size),
                                           dtype=torch.float32,
                                           device=self.gpu_device)

            self.attention_mask = torch.ones((batch_size, seq_length),
                                             dtype=torch.float32,
                                             device=self.gpu_device)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        def test_bertattention(self):
            self.num_iter = 50
            # Torch
            model = lambda: self.torch_attention(self.input_tensor, self.
                                                 attention_mask)
            torch_attention_result = self.run_model("Torch", model,
                                                    self.num_iter)

            ft_self_attention_result = self.ft_attention(
                self.input_tensor, self.attention_mask)

            ft_torch = 0.
            for it in range(self.num_iter):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                ft_self_attention_result = self.ft_attention(
                    self.input_tensor, self.attention_mask)
                end.record()
                torch.cuda.synchronize()
                ft_torch += start.elapsed_time(end)

            ft_torch /= 1e3
            print(
                f"BertAttention \"({batch_size}, {seq_length})\", FT QPS, {self.num_iter / ft_torch}, Elapse {ft_torch / self.num_iter}"
            )
            print(
                "max diff: ",
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              ft_self_attention_result)))
            self.assertTrue(
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              ft_self_attention_result)) < 1e-4)

        def run_model(self, model_name, model, num_iter=50):
            # warmup
            model()
            torch_elapsed = 0.
            for it in range(num_iter):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                torch_attention_result = model()
                end.record()
                torch.cuda.synchronize()
                torch_elapsed += start.elapsed_time(end)
            torch_elapsed /= 1e3
            # rescale to second
            print("BertAttention\"({}, {})\" {} QPS, {}, Elapse{}".format(
                batch_size, seq_length, model_name, num_iter / torch_elapsed,
                torch_elapsed / num_iter))
            return torch_attention_result

    globals()[f"TestBertAtt{batch_size}_{seq_length:3}"] = TestBertAttention


for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
