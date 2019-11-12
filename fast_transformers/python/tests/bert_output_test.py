import unittest

import contexttimer
import onnx
import torch
import torch.jit
import torch.onnx
import torch.utils.dlpack as dlpack
from onnxruntime.backend import backend
from transformers import BertTokenizer
from transformers.modeling_bert import BertConfig, BertOutput

import fast_transformers


def _(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))


def create_shape_test(batch_size: int, seq_length: int):
    class TestBertOut(unittest.TestCase):
        def setUp(self) -> None:
            fast_transformers.auto_init_blas()
            torch.set_grad_enabled(False)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.cfg = BertConfig(vocab_size_or_config_json_file=self.tokenizer.vocab_size)
            self.intermediate_size = self.cfg.intermediate_size  # 3072;
            self.hidden_size = self.cfg.hidden_size  # 768
            self.torch_bertout = BertOutput(self.cfg)
            self.torch_bertout.eval()

            bertout_params = {k: v for k, v in
                              self.torch_bertout.named_parameters()}

            self.ft_bertout = fast_transformers.BertOutput(
                _(bertout_params["dense.weight"]),
                _(bertout_params["dense.bias"]),
                _(bertout_params["LayerNorm.weight"]),
                _(bertout_params["LayerNorm.bias"])
            )

            self.intermediate_output = torch.rand(size=(batch_size, seq_length, self.intermediate_size),
                                                  dtype=torch.float32)
            self.attention_output = torch.rand(size=(batch_size, seq_length, self.hidden_size),
                                               dtype=torch.float32)

            self.jit_bertout = torch.jit.trace(self.torch_bertout,
                                               example_inputs=[self.intermediate_output, self.attention_output])

            torch.onnx.export(self.torch_bertout, (self.intermediate_output, self.attention_output), 'bertout.onnx')

            if not backend.supports_device('MKL-DNN'):
                self.onnx_bertout = backend.prepare(onnx.load('bertout.onnx'), device="CPU")
                print("Using CPU ONNX")
            else:
                self.onnx_bertout = backend.prepare(onnx.load('bertout.onnx'), device="MKL-DNN")
                print("Using MKL-DNN ONNX")

        def test_bertout(self):
            with open(f"bert_output_qps_{batch_size}_{seq_length:03}.txt", "w") as of:
                num_steps = 100
                torch_result = self.torch_bertout(self.intermediate_output, self.attention_output)
                with contexttimer.Timer() as t:
                    for it in range(num_steps):
                        torch_result = self.torch_bertout(self.intermediate_output, self.attention_output)

                print(f"BertOut({batch_size}, {seq_length:03}) Torch QPS {num_steps / t.elapsed}", file=of)

                with contexttimer.Timer() as t:
                    for it in range(num_steps):
                        self.jit_bertout(self.intermediate_output, self.attention_output)
                print(f'BertOut({batch_size}, {seq_length:03}) Jit QPS {num_steps / t.elapsed}', file=of)

                onnx_input_feeds = [self.intermediate_output.numpy(), self.attention_output.numpy()]
                self.onnx_bertout.run(inputs=onnx_input_feeds)

                with contexttimer.Timer() as t:
                    for it in range(num_steps):
                        self.onnx_bertout.run(inputs=onnx_input_feeds)
                print(f'BertOut({batch_size}, {seq_length:03}) ONNX QPS {num_steps / t.elapsed}', file=of)

                ft_result = dlpack.from_dlpack(
                    self.ft_bertout(_(self.intermediate_output), _(self.attention_output)).to_dlpack())
                with contexttimer.Timer() as t:
                    for it in range(num_steps):
                        ft_result = dlpack.from_dlpack(
                            self.ft_bertout(_(self.intermediate_output), _(self.attention_output)).to_dlpack())

                print(f"BertOut({batch_size}, {seq_length:03}) FastTransform QPS {num_steps / t.elapsed}", file=of)
                self.assertTrue( torch.max(torch.abs(torch_result - ft_result)) < 1e-4 )

    TestBertOut.__name__ = f"TestBertOut_BatchSize_{batch_size}_SeqLen_{seq_length}"

    globals()[TestBertOut.__name__] = TestBertOut

    return TestBertOut


TestCases = [create_shape_test(batch_size=batch_size, seq_length=seq_length) for seq_length in
             (20, 40, 60, 80, 100, 120) for batch_size in (1, 2)]

# TestBertOut = create_shape_test(batch_size=1, seq_length=20)

if __name__ == '__main__':
    # print(TestBertOut)
    unittest.main()
