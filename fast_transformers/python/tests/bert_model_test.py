import unittest
import os
import contexttimer
import torch
from transformers import BertTokenizer
from transformers.modeling_bert import BertModel, BertConfig
import numpy
import fast_transformers


class TestBertModel(unittest.TestCase):
    def setUp(self) -> None:
        model_id = os.path.join(os.path.dirname(__file__), 'test-model')
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        self.tokenizer = BertTokenizer.from_pretrained(model_id)
        self.torch_model = BertModel.from_pretrained(model_id)
        self.torch_model.eval()
        self.ft_model = fast_transformers.BertModel.from_pretrained(model_id)

    def test_bert_model(self):
        num_iter = 100
        input_ids = self.tokenizer.encode('测试一下bert模型的性能和精度是不是符合要求?')
        input_ids = torch.tensor([input_ids], dtype=torch.long)

        self.torch_model(input_ids)
        with contexttimer.Timer() as t:
            for it in range(num_iter):
                torch_result = self.torch_model(input_ids)
        print(f'BertModel Plain PyTorch QPS {num_iter / t.elapsed}')
        ft_result = self.ft_model(input_ids)
        with contexttimer.Timer() as t:
            for it in range(num_iter):
                ft_result = self.ft_model(input_ids)

        print(f'BertModel FastTransform QPS {num_iter / t.elapsed}')
        torch_result = (torch_result[0][:, 0]).numpy()
        ft_result = ft_result.numpy()

        # diff is 0.003478855
        # print("max diff is ", numpy.max(numpy.abs(torch_result - ft_result)))
        self.assertTrue(
            numpy.allclose(torch_result, ft_result, atol=5e-3, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
