from transformers.modeling_bert import BertEmbeddings, BertConfig
import unittest
import fast_transformers
import torch
import torch.utils.dlpack as dlpack
from transformers import BertTokenizer


def _(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))


class TestBertEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        fast_transformers.auto_init_blas()
        torch.set_grad_enabled(False)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        cfg = BertConfig(vocab_size_or_config_json_file=self.tokenizer.vocab_size)
        self.torch_embedding = BertEmbeddings(cfg)
        params = {k: _(v) for k, v in
                  self.torch_embedding.named_parameters()}
        self.torch_embedding.eval()

        self.ft_embedding = fast_transformers.BERTEmbedding(params['word_embeddings.weight'],
                                                            params['position_embeddings.weight'],
                                                            params['token_type_embeddings.weight'],
                                                            params['LayerNorm.weight'],
                                                            params['LayerNorm.bias'],
                                                            cfg.hidden_dropout_prob)

    def test_embedding(self):
        input_ids = torch.tensor(self.tokenizer.encode("这是测试数据?"), dtype=torch.long).reshape((1, -1))
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        token_type_ids = torch.zeros_like(input_ids)

        torch_result = self.torch_embedding(input_ids, token_type_ids, position_ids)
        ft_result = dlpack.from_dlpack(self.ft_embedding(_(input_ids), _(position_ids), _(token_type_ids)).to_dlpack())
        self.assertTrue(torch.max(torch_result - ft_result) < 1e-5)


if __name__ == '__main__':
    unittest.main()
