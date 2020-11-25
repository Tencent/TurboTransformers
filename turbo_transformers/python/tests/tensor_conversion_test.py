import torch
import torch.utils.dlpack as dlpack
import unittest
from turbo_transformers.layers.modeling_bert import convert2tt_tensor
import turbo_transformers


class TestDLPack(unittest.TestCase):
    def check_dlpack(self, use_cuda):
        test_device = torch.device('cuda:0') if use_cuda else \
            torch.device('cpu:0')
        if not use_cuda:
            torch.set_num_threads(1)

        a = torch.rand(size=(4, 3), dtype=torch.float32, device=test_device)
        tensor = convert2tt_tensor(a)
        self.assertIsNotNone(tensor)
        b = dlpack.from_dlpack(tensor.to_dlpack())

        self.assertTrue(a.equal(b))
        self.assertTrue(b.cpu().equal(a.cpu()))

    def test_dlpack(self):
        self.check_dlpack(use_cuda=False)
        if torch.cuda.is_available() and \
            turbo_transformers.config.is_compiled_with_cuda():
            self.check_dlpack(use_cuda=True)


if __name__ == '__main__':
    unittest.main()
