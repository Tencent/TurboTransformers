import unittest
import os
import contexttimer
import torch
import torch.jit
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertEncoder

import fast_transformers


def create_test(batch_size, seq_length):
    if not torch.cuda.is_available():
        return

    class TestBertEncoder(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            torch.set_num_threads(1)

            self.test_device = torch.device('cuda:0')
            self.device = "GPU"

            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size,
                attention_probs_dropout_prob=0.0,
                hidden_dropout_prob=0.0,
                num_hidden_layers=2)

            self.torch_encoder_layer = BertEncoder(self.cfg)
            self.torch_encoder_layer.eval()
            if torch.cuda.is_available():
                self.torch_encoder_layer.to(self.test_device)

            self.batch_size = 1
            self.seq_length = 40
            self.hidden_size = self.cfg.hidden_size
            self.input_tensor = torch.rand(size=(self.batch_size,
                                                 self.seq_length,
                                                 self.hidden_size),
                                           dtype=torch.float32,
                                           device=self.test_device)

            self.attention_mask = torch.ones(
                (self.batch_size, self.seq_length),
                dtype=torch.float32,
                device=self.test_device)
            self.attention_mask = self.attention_mask[:, None, None, :]
            self.attention_mask = (1.0 - self.attention_mask) * -10000.0

            self.ft_bert_encoder = fast_transformers.BertEncoder.from_torch(
                self.torch_encoder_layer)

        def test_bert_encoder(self):
            num_iter = 100

            ft_bert_layer_result = self.ft_bert_encoder(
                self.input_tensor, self.attention_mask)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with contexttimer.Timer() as t:
                ft_bert_layer_result = None
                for it in range(num_iter):
                    ft_bert_layer_result = self.ft_bert_encoder(
                        self.input_tensor,
                        self.attention_mask,
                        output=ft_bert_layer_result,
                        return_type=fast_transformers.ReturnType.
                        FAST_TRANSFORMERS)
            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                # in ms, rescale to sec
                ft_elapsed = start.elapsed_time(end) / 1e3
                ft_qps = num_iter / ft_elapsed
                ft_time = ft_elapsed / num_iter
            else:
                ft_qps = num_iter / t.elapsed
                ft_time = t.elapsed / num_iter
            ft_bert_layer_result = self.ft_bert_encoder(
                self.input_tensor, self.attention_mask)

            print(
                f"BertEncoder \"({batch_size},{seq_length:03})\" {self.device} FastTransform QPS,  {ft_qps}, time, {ft_time}"
            )

            torch_bert_layer_result = self.torch_encoder_layer(
                self.input_tensor, self.attention_mask,
                [None] * self.cfg.num_hidden_layers)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_bert_layer_result = self.torch_encoder_layer(
                        self.input_tensor, self.attention_mask,
                        [None] * self.cfg.num_hidden_layers)

            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                torch_elapsed = start.elapsed_time(end) / 1e3
                torch_qps = num_iter / torch_elapsed
                torch_time = torch_elapsed / num_iter
            else:
                torch_qps = num_iter / t.elapsed
                torch_time = t.elapsed / num_iter

            print(
                f"BertEncoder \"({batch_size},{seq_length:03})\" {self.device} Torch QPS,  {torch_qps}, time, {torch_time}"
            )

            diff = torch.abs(torch_bert_layer_result[0] - ft_bert_layer_result)
            self.assertTrue(torch.max(diff) < 1e-3)

    globals(
    )[f"TestBertEncoder_{batch_size}_{seq_length:03}"] = TestBertEncoder


for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)
if __name__ == '__main__':
    unittest.main()
