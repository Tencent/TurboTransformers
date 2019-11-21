import unittest
import fast_transformers
import torch
import numpy as np


def pooling(input, pool_type):
    if pool_type == "First":
        return input[:, 0, :]
    elif pool_type == "Last":
        seq_len = input.shape[1]
        return input[:, seq_len - 1, :]
    elif pool_type == "Mean":
        return np.mean(input, axis=1)
    elif pool_type == "Max":
        return np.max(input, axis=1)
    else:
        raise "{} is not support.".format(pool_type)


def create_test_seq_pool(batch_size: int, seq_length: int, pool_type: str):
    class TestSequencePool(unittest.TestCase):
        def setUp(self) -> None:
            torch.set_grad_enabled(False)
            hidden_size = 50
            self.input = np.random.random(
                (batch_size, seq_length, hidden_size)).astype("float32")
            self.seq_pool = fast_transformers.SequencePool(pool_type)

        def test_seq_pool(self):
            np_result = pooling(self.input, pool_type)
            ft_result = self.seq_pool(torch.tensor(self.input))
            self.assertTrue(
                np.max(np.abs(np_result - ft_result.numpy())) < 1e-3)

    globals(
    )[f"TestSequencePool{batch_size}_{seq_length:03}_{pool_type}"] = TestSequencePool


for batch_size in [1, 5]:
    for seq_length in [5, 8]:
        for pool_type in ["Mean", "Max", "First", "Last"]:
            create_test_seq_pool(batch_size, seq_length, pool_type)

if __name__ == '__main__':
    unittest.main()
