import unittest

import contexttimer
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import fast_transformers


def create_test_bert_emb(batch_size: int, seq_length: int):
    class TestBertEmbedding(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            torch.set_num_threads(1)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)
            self.torch_model = BertModel(cfg)
            self.torch_model.eval()
            self.ft_embedding = fast_transformers.BertModel.from_torch(
                self.torch_model)

        def test_embedding(self):
            num_iter = 100
            input_ids = torch.randint(low=0,
                                      high=self.tokenizer.vocab_size - 1,
                                      size=(batch_size, seq_length),
                                      dtype=torch.long)
            # warming up.
            self.torch_model(input_ids)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_result = self.torch_model(input_ids)
            print(
                f'BertEmb({batch_size}, {seq_length:03}) Plain PyTorch QPS {num_iter / t.elapsed}'
            )
            ft_result = self.ft_embedding(input_ids)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    ft_result = self.ft_embedding(input_ids)

            print(
                f'BertEmb({batch_size}, {seq_length:03}) FastTransform QPS {num_iter / t.elapsed}'
            )
            torch_result = (torch_result[0][:, 0]).numpy()
            ft_result = ft_result.numpy()
            self.assertTrue(
                numpy.allclose(torch_result, ft_result, atol=1e-3, rtol=1e-4))

    globals(
    )[f"TestBertEmbedding{batch_size}_{seq_length:03}"] = TestBertEmbedding


create_test_bert_emb(2, 40)
if __name__ == '__main__':
    unittest.main()
