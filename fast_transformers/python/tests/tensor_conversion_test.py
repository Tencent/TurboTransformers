import torch
import torch.utils.dlpack as dlpack
import unittest
from utils import convert2ft_tensor

class TestDLPack(unittest.TestCase):
    def test_dlpack(self):
        a = torch.rand(size=(34, 73), dtype=torch.float32)
        tensor = convert2ft_tensor(a)
        self.assertIsNotNone(tensor)
        b = dlpack.from_dlpack(tensor.to_dlpack())
        self.assertTrue(a.equal(b))


if __name__ == '__main__':
    unittest.main()
