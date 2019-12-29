import unittest

import contexttimer
import torch
import torch.jit
import torch.onnx
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertLayer
import os

import fast_transformers


class TestBertLayer(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_grad_enabled(False)
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(os.path.dirname(__file__), 'test-model'))
        self.cfg = BertConfig(
            vocab_size_or_config_json_file=self.tokenizer.vocab_size,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0)

        self.torch_bert_layer = BertLayer(self.cfg)
        self.torch_bert_layer.eval()

        self.batch_size = 1
        self.seq_length = 400
        self.hidden_size = self.cfg.hidden_size
        self.input_tensor = torch.rand(size=(self.batch_size, self.seq_length,
                                             self.hidden_size),
                                       dtype=torch.float32)

        self.attention_mask = torch.ones((self.batch_size, self.seq_length),
                                         dtype=torch.long)
        self.attention_mask = self.attention_mask[:, None, None, :]
        self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        self.ft_bert_layer = fast_transformers.BertLayer.from_torch(
            self.torch_bert_layer)

    def test_bert_layer(self):
        self.num_iter = 10

        torch_bert_layer_result = self.torch_bert_layer(
            self.input_tensor, self.attention_mask)
        with contexttimer.Timer() as t:
            for it in range(self.num_iter):
                torch_bert_layer_result = self.torch_bert_layer(
                    self.input_tensor, self.attention_mask)

        print(f"BertLayer Torch QPS, {self.num_iter / t.elapsed}, ",
              f"Time Cost, {t.elapsed / self.num_iter}")

        with fast_transformers.gperf_guard("bert_layer.gperf"):
            ft_bert_layer_result = self.ft_bert_layer(self.input_tensor,
                                                      self.attention_mask)
        with contexttimer.Timer() as t:
            for it in range(self.num_iter):
                ft_bert_layer_result = self.ft_bert_layer(
                    self.input_tensor, self.attention_mask)

        print(
            f"fast_transformer BertLayer Torch QPS, {self.num_iter / t.elapsed}, ",
            f"Time Cost, {t.elapsed / self.num_iter}")
        self.assertTrue(
            torch.max(
                torch.abs(torch_bert_layer_result[0] -
                          ft_bert_layer_result)) < 1e-3)


for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
