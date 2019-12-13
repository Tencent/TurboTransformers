import unittest

import contexttimer
import torch
import fast_transformers
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertIntermediate
import numpy
import os


def create_test(batch_size, seq_length):
    class TestBertIntermediate(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_num_threads(1)
            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)

            self.torch_intermediate = BertIntermediate(self.cfg)
            self.torch_intermediate.eval()

            self.ft_intermediate = fast_transformers.BertIntermediate.from_torch(
                self.torch_intermediate)

        def test_intermediate(self):
            num_iter = 2000
            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32)

            ft_result = self.ft_intermediate(input_tensor)

            with contexttimer.Timer() as t:
                ft_result = None
                for it in range(num_iter):
                    ft_result = self.ft_intermediate(
                        input_tensor,
                        output=ft_result,
                        return_type=fast_transformers.ReturnType.
                        FAST_TRANSFORMERS)

            ft_result = self.ft_intermediate(input_tensor)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) FastTransform QPS,  {num_iter / t.elapsed}, time, {t.elapsed / num_iter}"
            )
            self.torch_intermediate(input_tensor)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_result = self.torch_intermediate(input_tensor)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) Torch QPS,  {num_iter / t.elapsed}, time, {t.elapsed / num_iter}"
            )
            torch_result = torch_result.numpy()
            ft_result = ft_result.numpy()

            self.assertTrue(
                numpy.allclose(torch_result, ft_result, rtol=1e-4, atol=1e-3))

    globals(
    )[f"TestBertIntermediate_{batch_size}_{seq_length:03}"] = TestBertIntermediate


create_test(1, 40)
if __name__ == '__main__':
    unittest.main()
