import unittest

import contexttimer
import fast_transformers
import onnxruntime.backend as backend
import torch
import torch.jit
import torch.onnx
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertIntermediate


def create_test(batch_size, seq_length):
    class TestBertIntermediate(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.cfg = BertConfig(
                vocab_size_or_config_json_file=self.tokenizer.vocab_size)

            self.torch_intermediate = BertIntermediate(self.cfg)
            self.torch_intermediate.eval()

            self.jit_intermediate = torch.jit.trace(
                self.torch_intermediate,
                example_inputs=[
                    torch.rand(size=(batch_size, seq_length,
                                     self.cfg.hidden_size),
                               dtype=torch.float32)
                ])

            torch.onnx.export(self.torch_intermediate, (torch.rand(
                size=(batch_size, seq_length, self.cfg.hidden_size),
                dtype=torch.float32), ), "bert-intermediate.onnx")

            if not backend.supports_device('MKL-DNN'):
                self.onnx_intermedia = backend.prepare(
                    "bert-intermediate.onnx", 'CPU')
            else:
                self.onnx_intermedia = backend.prepare(
                    "bert-intermediate.onnx", 'MKL-DNN')
            self.ft_intermediate = fast_transformers.BertIntermediate.from_torch(
                self.torch_intermediate)

        def test_intermediate(self):
            num_iter = 100
            hidden_size = self.cfg.hidden_size
            input_tensor = torch.rand(size=(batch_size, seq_length,
                                            hidden_size),
                                      dtype=torch.float32)
            torch_result = self.torch_intermediate(input_tensor)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    torch_result = self.torch_intermediate(input_tensor)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) Torch QPS,  {num_iter / t.elapsed}, time, {t.elapsed / num_iter}"
            )

            jit_results = self.jit_intermediate(input_tensor)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    jit_results = self.jit_intermediate(input_tensor)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) JIT QPS,  {num_iter / t.elapsed}, time, {t.elapsed / num_iter}"
            )

            onnx_inputs = [input_tensor.numpy()]
            onnx_results = self.onnx_intermedia.run(onnx_inputs)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    onnx_results = self.onnx_intermedia.run(onnx_inputs)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) ONNX QPS,  {num_iter / t.elapsed}, time, {t.elapsed / num_iter}"
            )

            ft_result = self.ft_intermediate(input_tensor)
            with contexttimer.Timer() as t:
                for it in range(num_iter):
                    ft_result = self.ft_intermediate(input_tensor)

            print(
                f"BertIntermediate ({batch_size},{seq_length:03}) FastTransform QPS,  {num_iter / t.elapsed}, time, {t.elapsed / num_iter}"
            )
            self.assertTrue(
                torch.max(torch.abs(torch_result -
                                    torch.tensor(onnx_results))) < 0.001)
            self.assertTrue(
                torch.max(torch.abs(torch_result - jit_results)) < 0.001)
            self.assertTrue(
                torch.max(torch.abs(torch_result - ft_result)) < 0.001)

    globals(
    )[f"TestBertIntermediate_{batch_size}_{seq_length:03}"] = TestBertIntermediate


create_test(1, 32)

# for batch_size in [1, 16]:
#     for seq_length in [16, 32, 64, 128]:
#         create_test(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
