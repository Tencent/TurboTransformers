import unittest

import contexttimer
import torch
import torch.jit
import torch.onnx
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertEncoder

import fast_transformers


class TestBertEncoder(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.cfg = BertConfig(
            vocab_size_or_config_json_file=self.tokenizer.vocab_size,
            attention_probs_dropout_prob=0.0,
            hidden_dropout_prob=0.0,
            num_hidden_layers=2)

        self.torch_encoder_layer = BertEncoder(self.cfg)
        self.torch_encoder_layer.eval()

        self.batch_size = 1
        self.seq_length = 40
        self.hidden_size = self.cfg.hidden_size
        self.input_tensor = torch.rand(size=(self.batch_size, self.seq_length,
                                             self.hidden_size),
                                       dtype=torch.float32)

        self.attention_mask = torch.ones((self.batch_size, self.seq_length),
                                         dtype=torch.long)
        self.attention_mask = self.attention_mask[:, None, None, :]
        self.attention_mask = (1.0 - self.attention_mask) * -10000.0

        self.ft_bert_encoder = fast_transformers.BertEncoder.from_torch(
            self.torch_encoder_layer)

    def test_bert_encoder(self):
        self.num_iter = 100

        ft_bert_layer_result = self.ft_bert_encoder(self.input_tensor,
                                                    self.attention_mask)
        with contexttimer.Timer() as t:
            ft_bert_layer_result = None
            for it in range(self.num_iter):
                ft_bert_layer_result = self.ft_bert_encoder(
                    self.input_tensor,
                    self.attention_mask,
                    output=ft_bert_layer_result,
                    return_type=fast_transformers.ReturnType.FAST_TRANSFORMERS)
        ft_bert_layer_result = self.ft_bert_encoder(self.input_tensor,
                                                    self.attention_mask)
        print(
            f"fast_transformer BertEncoder Torch QPS, {self.num_iter / t.elapsed}, ",
            f"Time Cost, {t.elapsed / self.num_iter}")

        torch_bert_layer_result = self.torch_encoder_layer(
            self.input_tensor, self.attention_mask,
            [None] * self.cfg.num_hidden_layers)
        with contexttimer.Timer() as t:
            for it in range(self.num_iter):
                torch_bert_layer_result = self.torch_encoder_layer(
                    self.input_tensor, self.attention_mask,
                    [None] * self.cfg.num_hidden_layers)

        print(f"BertEncoder Torch QPS, {self.num_iter / t.elapsed}, ",
              f"Time Cost, {t.elapsed / self.num_iter}")

        diff = torch.abs(torch_bert_layer_result[0] - ft_bert_layer_result)
        # print(diff)
        self.assertTrue(torch.max(diff) < 1e-3)


if __name__ == '__main__':
    unittest.main()
