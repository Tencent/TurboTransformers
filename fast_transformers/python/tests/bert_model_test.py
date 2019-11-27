import unittest

import contexttimer
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import fast_transformers

batch_size = 2
seq_length = 40


class TestBertModel(unittest.TestCase):
    def setUp(self) -> None:

        model_id = "bert-base-chinese"
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.torch_model = BertModel.from_pretrained(model_id)
        self.torch_model.eval()
        self.ft_embedding = fast_transformers.BertModel.from_pretrained(
            model_id)

    def test_bert_model(self):
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


if __name__ == '__main__':
    unittest.main()
