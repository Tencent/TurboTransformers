import fast_transformers

import unittest
import contexttimer
import torch
import torch.jit
import torch.onnx
from transformers.modeling_bert import BertConfig, BertAttention
from transformers import BertTokenizer
import os

fname = "ft_attention.txt"


def create_test(batch_size, seq_length):
    class TestBertAttention(unittest.TestCase):
        def setUp(self) -> None:
            if not torch.cuda.is_available(
            ) or not fast_transformers.config.is_with_cuda():
                torch.set_num_threads(1)
                self.test_device = torch.device('cpu')
                self.device = "CPU"
            else:
                self.test_device = torch.device('cuda:0')
                self.device = "GPU"
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
            if torch.cuda.is_available():
                self.torch_attention.to(self.test_device)

            # Get FT Attention
            num_attention_heads = self.cfg.num_attention_heads
            self.ft_attention = fast_transformers.BertAttention.from_torch(
                self.torch_attention)

            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(batch_size, seq_length,
                                                 self.hidden_size),
                                           dtype=torch.float32,
                                           device=self.test_device)

            self.attention_mask = torch.ones((batch_size, seq_length),
                                             dtype=torch.float32,
                                             device=self.test_device)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        def test_bertattention(self):
            num_iter = 50
            # Torch
            model = lambda: self.torch_attention(self.input_tensor, self.
                                                 attention_mask)
            torch_attention_result = self.run_model("Torch", model, num_iter)

            ft_self_attention_result = self.ft_attention(
                self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                ft_elapsed = 1e-3
                ft_result = None
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    ft_self_attention_result = self.ft_attention(
                        self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                # in ms, rescale to sec
                ft_elapsed = start.elapsed_time(end) / 1e3

            ft_qps = 0
            ft_time = 0
            if torch.cuda.is_available():
                ft_qps = num_iter / ft_elapsed
                ft_time = ft_elapsed / num_iter
            else:
                ft_qps = num_iter / t.elapsed
                ft_time = t.elapsed / num_iter

            print(
                f"BertAttention \"({batch_size},{seq_length:03})\" {self.device} FastTransform QPS,  {ft_qps}, time, {ft_time}"
            )
            self.assertTrue(
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              ft_self_attention_result)) < 1e-4)

            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {self.torch_qps}, {ft_qps}\n"
                )

        def run_model(self, model_name, model, num_iter=50):
            # warmup
            model()

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_attention_result = model()

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                torch_elapsed = start.elapsed_time(end) / 1e3

            if torch.cuda.is_available():
                self.torch_qps = num_iter / torch_elapsed
                self.torch_time = torch_elapsed / num_iter
            else:
                self.torch_qps = num_iter / t.elapsed
                self.torch_time = t.elapsed / num_iter

            print(
                f"BertAttention \"({batch_size},{seq_length:03})\" {self.device} Torch QPS,  {self.torch_qps}, time, {self.torch_time}"
            )
            return torch_attention_result

    globals()[f"TestBertAtt{batch_size}_{seq_length:3}"] = TestBertAttention


with open(fname, "w") as fh:
    fh.write(", torch, fast_transformers\n")
for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
