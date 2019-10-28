import torch
import torch.utils.dlpack as dlpack
import fast_transformers
import unittest


class TestDLPack(unittest.TestCase):
    def setUp(self) -> None:
        fast_transformers.auto_init_blas()

    def test_dlpack(self):
        a = torch.rand(size=(34, 73), dtype=torch.float32)
        tensor = fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(a))
        self.assertIsNotNone(tensor)
        b = dlpack.from_dlpack(tensor.to_dlpack())
        self.assertTrue(a.equal(b))


if __name__ == '__main__':
    unittest.main()
