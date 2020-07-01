
import turbo_transformers

import unittest
import sys
import torch
from transformers.modeling_albert import AlbertConfig, AlbertAttention
import os

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_attention.txt"


def create_test(batch_size, seq_length):
    class TestAlbertAttention(unittest.TestCase):
        def init_data(self, use_cuda):
            test_device = torch.device('cuda:0') if use_cuda else \
                   torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            cfg = AlbertConfig(attention_probs_dropout_prob=0.0,
                             hidden_dropout_prob=0.0)
            torch_attention = AlbertAttention(cfg)
            torch_attention.eval()
            if use_cuda:
                torch_attention.to(test_device)

            # Get FT Attention
            turbo_attention = turbo_transformers.AlbertAttention.from_torch(
                torch_attention)
            hidden_size = cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32,
                                      device=test_device)
            attention_mask = torch.ones((batch_size, seq_length),
                                        dtype=torch.float32,
                                        device=test_device)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            return torch_attention, turbo_attention, input_tensor, attention_mask

        def check_torch_and_turbo(self, use_cuda, num_iter=2):
            torch_attention, turbo_attention, input_tensor, attention_mask = \
                self.init_data(use_cuda)
            device = "GPU" if use_cuda else "CPU"
            torch_model = lambda: torch_attention(input_tensor, attention_mask)
            torch_attention_result, torch_qps, torch_time_consume = \
                test_helper.run_model(torch_model, use_cuda, num_iter)
            print(
                f"AlbertAttention \"({batch_size},{seq_length:03})\" ",
                f"{device} Torch QPS, {torch_qps}, time, {torch_time_consume}")

            turob_model = lambda: turbo_attention(input_tensor, attention_mask)
            turbo_self_attention_result, turbo_qps, turbo_time_consume = \
                test_helper.run_model(turob_model, use_cuda,
                                      num_iter)
            print(
                f"AlbertAttention \"({batch_size},{seq_length:03})\" ",
                f" {device} Turbo QPS, {turbo_qps}, time, {turbo_time_consume}"
            )

            self.assertTrue(
                torch.max(
                    torch.abs(torch_attention_result[0] -
                              turbo_self_attention_result)) < 1e-3
                if use_cuda else 1e-4)
            with open(fname, "a") as fh:
                fh.write(
                    f"\"({batch_size},{seq_length:03})\", {torch_qps}, {turbo_qps}\n"
                )

        def test_Albert_attention(self):
            self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals()[f"TestAlbertAtt{batch_size}_{seq_length:3}"] = TestAlbertAttention


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")
for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
