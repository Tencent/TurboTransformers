import unittest

import contexttimer
import time
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
            if torch.cuda.is_available():
                self.test_device = torch.device('cuda:0')
            else:
                self.test_device = torch.device('cpu')

            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(os.path.dirname(__file__), 'test-model'))
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)

            self.torch_intermediate = BertIntermediate(self.cfg)
            if torch.cuda.is_available():
                self.torch_intermediate.to(self.test_device)
            self.torch_intermediate.eval()

            self.ft_intermediate = fast_transformers.BertIntermediate.from_torch(
                self.torch_intermediate)

        def test_intermediate(self):
            num_iter = 2000
            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32,
                                      device=self.test_device)

            #warmup
            ft_result = self.ft_intermediate(input_tensor)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            ft_elapsed = 0.
            ft_result = None
            start.record()
            for it in range(num_iter):
                ft_result = self.ft_intermediate(
                    input_tensor,
                    output=ft_result,
                    return_type=fast_transformers.ReturnType.FAST_TRANSFORMERS)
            end.record()
            torch.cuda.synchronize()
            # in ms, rescale to sec
            ft_elapsed = start.elapsed_time(end) / 1e3

            #get torch result
            ft_result = self.ft_intermediate(input_tensor)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) FastTransform QPS,  {num_iter / ft_elapsed}, time, {ft_elapsed / num_iter}"
            )

            #warmup
            torch_result = self.torch_intermediate(input_tensor)
            torch_elapsed = 0.

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for it in range(num_iter):
                torch_result = self.torch_intermediate(input_tensor)
            end.record()
            torch.cuda.synchronize()
            torch_elapsed = start.elapsed_time(end) / 1e3
            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) Torch QPS,  {num_iter / torch_elapsed}, time, {torch_elapsed / num_iter}"
            )
            torch_result = torch_result.cpu().numpy()
            ft_result = ft_result.cpu().numpy()
            #print("diff ", numpy.max(torch_result - ft_result))

            self.assertTrue(
                numpy.allclose(torch_result, ft_result, rtol=1e-4, atol=1e-3))

    globals(
    )[f"TestBertIntermediate_{batch_size}_{seq_length:03}"] = TestBertIntermediate


for batch_size in [1, 2]:
    for seq_length in [10, 16, 20, 24, 40, 48, 60, 64, 80, 100, 120, 128]:
        create_test(batch_size, seq_length)
if __name__ == '__main__':
    unittest.main()
