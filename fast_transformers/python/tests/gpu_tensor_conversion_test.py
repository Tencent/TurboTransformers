import torch
import torch.utils.dlpack as dlpack
import unittest
from fast_transformers.layers.modeling_bert import convert2ft_tensor


class TestDLPack(unittest.TestCase):
    def test_dlpack(self):
        if torch.cuda.is_available():
            self.test_device = torch.device('cuda:0')
        else:
            self.test_device = torch.device('cpu')

        a = torch.rand(size=(4, 3),
                       dtype=torch.float32,
                       device=self.test_device)
        tensor = convert2ft_tensor(a)
        self.assertIsNotNone(tensor)
        b = dlpack.from_dlpack(tensor.to_dlpack())

        self.assertTrue(a.equal(b))
        self.assertTrue(b.cpu().equal(a.cpu()))


if __name__ == '__main__':
    unittest.main()
