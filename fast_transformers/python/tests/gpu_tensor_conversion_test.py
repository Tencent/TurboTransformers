import torch
import torch.utils.dlpack as dlpack
import unittest
from fast_transformers.layers.modeling_bert import convert2ft_tensor


class TestDLPack(unittest.TestCase):
    def test_dlpack(self):
        cuda = torch.device('cuda:0')
        a = torch.rand(size=(4, 3), dtype=torch.float32, device=cuda)
        tensor = convert2ft_tensor(a)
        self.assertIsNotNone(tensor)
        b = dlpack.from_dlpack(tensor.to_dlpack())

        self.assertTrue(a.equal(b))
        self.assertTrue(b.cpu().equal(a.cpu()))


if __name__ == '__main__':
    unittest.main()
