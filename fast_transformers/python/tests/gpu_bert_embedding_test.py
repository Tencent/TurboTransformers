import unittest

import contexttimer
import fast_transformers
import torch
import torch.jit
from transformers import BertTokenizer
from transformers.modeling_bert import BertEmbeddings, BertConfig
import os


def create_test_bert_emb(batch_size: int, seq_length: int):
    if not torch.cuda.is_available():
        return

    class TestBertEmbedding(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)
            self.torch_embedding = BertEmbeddings(cfg)
            self.test_device = torch.device('cuda:0')
            self.device = "GPU"
            self.torch_embedding.eval()

            if torch.cuda.is_available():
                self.torch_embedding.to(self.test_device)

            self.ft_embedding = fast_transformers.BertEmbeddings.from_torch(
                self.torch_embedding)

        def test_embedding(self):
            num_iter = 100
            input_ids = torch.randint(low=0,
                                      high=self.tokenizer.vocab_size - 1,
                                      size=(batch_size, seq_length),
                                      dtype=torch.long,
                                      device=self.test_device)
            position_ids = torch.arange(seq_length,
                                        dtype=torch.long,
                                        device=input_ids.device)
            # position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #will cause bug
            position_ids = position_ids.repeat(batch_size, 1)
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)

            # warming up.
            self.torch_embedding(input_ids, token_type_ids, position_ids)

            if torch.cuda.is_available():
                torch_elapsed = 0.
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    self.torch_embedding(input_ids, token_type_ids,
                                         position_ids)

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
                f"BertEmbeddings \"({batch_size},{seq_length:03})\" {self.device} Torch QPS,  {torch_qps}, time, {torch_time}"
            )
            torch_result = self.torch_embedding(input_ids, token_type_ids,
                                                position_ids)

            # warmup
            ft_result = self.ft_embedding(input_ids, position_ids,
                                          token_type_ids)

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                ft_elapsed = 0.
                start.record()

            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    ft_result = self.ft_embedding(input_ids, position_ids,
                                                  token_type_ids)

            ft_qps = 0
            ft_time = 0
            if torch.cuda.is_available():
                end.record()
                torch.cuda.synchronize()
                ft_elapsed = start.elapsed_time(end) / 1e3
                ft_qps = num_iter / ft_elapsed
                ft_time = ft_elapsed / num_iter
            else:
                ft_qps = num_iter / t.elapsed
                ft_time = t.elapsed / num_iter

            self.assertTrue(
                torch.max(torch.abs(torch_result - ft_result)) < 1e-5)
            print(
                f"BertEmbeddings\"({batch_size},{seq_length:03})\" {self.device} FastTransform QPS,  {ft_qps}, time, {ft_time}"
            )

    globals(
    )[f"TestBertEmbedding{batch_size}_{seq_length:03}"] = TestBertEmbedding


for batch_size in [1, 2]:
    for seq_length in [10, 20, 40, 80, 100, 120]:
        create_test_bert_emb(batch_size, seq_length)
# create_test_bert_emb(2, 10)
if __name__ == '__main__':
    unittest.main()
