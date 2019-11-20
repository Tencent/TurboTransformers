import unittest

import contexttimer
import fast_transformers
import onnxruntime.backend as backend
import torch
import torch.jit
import torch.onnx


def create_test_seq_pool(batch_size: int, seq_length: int):
    class TestSequencePool(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            hidden_size = 50
            self.input = torch.rand(size=(batch_size, seq_length, hidden_size))
            self.seq_pool = fast_transformers.SequencePool("Mean")

        def test_embedding(self):
            print(self.input)
            print(self.input.shape)
            ft_result = self.seq_pool(self.input)
            print(ft_result)
            print(ft_result.shape)

    globals(
    )[f"TestSequencePool{batch_size}_{seq_length:03}"] = TestSequencePool


for batch_size in [2, 4]:
    for seq_length in [5, 8]:
        create_test_seq_pool(batch_size, seq_length)

if __name__ == '__main__':
    unittest.main()
