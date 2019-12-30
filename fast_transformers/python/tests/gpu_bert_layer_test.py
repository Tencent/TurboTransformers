import unittest

import contexttimer
import torch
import torch.jit
import torch.onnx
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertLayer
import os

import fast_transformers


def create_test(batch_size, seq_length):
    class TestBertLayer(unittest.TestCase):
        def setUp(self) -> None:
            if torch.cuda.is_available():
                self.test_device = torch.device('cuda:0')
            else:
                self.test_device = torch.device('cpu')

            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size,
                attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0)

            self.torch_bert_layer = BertLayer(self.cfg)
            self.torch_bert_layer.eval()
            if torch.cuda.is_available():
                self.torch_bert_layer.to(self.test_device)

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

            self.ft_bert_layer = fast_transformers.BertLayer.from_torch(
                self.torch_bert_layer)

        def test_bert_layer(self):
            self.num_iter = 10

            torch_bert_layer_result = self.torch_bert_layer(
                self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch_elapsed = 0.
                start.record()

            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    torch_bert_layer_result = self.torch_bert_layer(
                        self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                torch_elapsed = start.elapsed_time(end) / 1e3

            print(f"BertLayer Torch QPS, {self.num_iter / t.elapsed}, ",
                  f"Time Cost, {t.elapsed / self.num_iter}")

            print(
                f"BertLayer Torch QPS event, {self.num_iter / torch_elapsed}, ",
                f"Time Cost, {torch_elapsed / self.num_iter}")

            with fast_transformers.gperf_guard("bert_layer.gperf"):
                ft_bert_layer_result = self.ft_bert_layer(
                    self.input_tensor, self.attention_mask)
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                ft_elapsed = 0.
                start.record()

            with contexttimer.Timer() as t:
                for it in range(self.num_iter):
                    ft_bert_layer_result = self.ft_bert_layer(
                        self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                ft_elapsed = start.elapsed_time(end) / 1e3

            print(
                f"fast_transformer BertLayer Torch QPS, {self.num_iter / t.elapsed}, ",
                f"Time Cost, {t.elapsed / self.num_iter}")

            if torch.cuda.is_available():
                print(
                    f"fast_transformer BertLayer Torch QPS event, {self.num_iter / ft_elapsed}, ",
                    f"Time Cost, {ft_elapsed / self.num_iter}")

            self.assertTrue(
                torch.max(
                    torch.abs(torch_bert_layer_result[0] -
                              ft_bert_layer_result)) < 1e-3)

            if torch.cuda.is_available():
                print(
                    f"BertLayer QPS event, \"({batch_size}, {seq_length})\", {self.num_iter / ft_elapsed}, {self.num_iter / torch_elapsed}"
                )

    globals()[f"TestBertLayer{batch_size}_{seq_length:03}"] = TestBertLayer


#for batch_size in [1, 20]:
#    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
#        create_test(batch_size, seq_length)

create_test(20, 40)

if __name__ == '__main__':
    unittest.main()
