from transformers.modeling_bert import BertEmbeddings, BertConfig
import unittest
import fast_transformers
import torch
import torch.utils.dlpack as dlpack
import torch.jit
import torch.onnx
from transformers import BertTokenizer
import contexttimer
import onnxruntime
import onnxruntime.backend as backend


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

        torch.onnx.export(self.torch_embedding, (
            torch.ones(size=(1, 7), dtype=torch.long), torch.ones(size=(1, 7), dtype=torch.long),
            torch.ones(size=(1, 7), dtype=torch.long)), f="bert-emb.onnx", output_names=['emb'])
        onnx_sess_opts = onnxruntime.SessionOptions()
        onnx_sess_opts.max_num_graph_transformation_steps = 10
        # onnx_sess_opts.set_graph_optimization_level(2)
        self.onnx_embedding = onnxruntime.InferenceSession("bert-emb.onnx", sess_options=onnx_sess_opts)
        backend.prepare(self.onnx_embedding, "CPU-MKL-DNN")

        self.torch_script_embedding = torch.jit.trace(
            self.torch_embedding, (torch.ones(size=(1, 7), dtype=torch.long), torch.ones(size=(1, 7), dtype=torch.long),
                                   torch.ones(size=(1, 7), dtype=torch.long)))

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
        # warming up.
        self.torch_embedding(input_ids, token_type_ids, position_ids)
        self.torch_script_embedding(input_ids, token_type_ids, position_ids)
        onnx_inputs = self.onnx_embedding.get_inputs()
        onnx_input_feeds = {
            onnx_inputs[0].name: input_ids.numpy(),
            onnx_inputs[1].name: token_type_ids.numpy(),
            onnx_inputs[2].name: position_ids.numpy()
        }
        self.onnx_embedding.run(output_names=['emb'],
                                input_feed=onnx_input_feeds)
        with contexttimer.Timer() as t:
            for it in range(100):
                torch_result = self.onnx_embedding.run(output_names=['emb'],
                                                       input_feed=onnx_input_feeds)
        print(f'ONNX (with mkl-dnn) time {t.elapsed}')

        with contexttimer.Timer() as t:
            for it in range(100):
                torch_result = self.torch_embedding(input_ids, token_type_ids, position_ids)
        print(f'Plain PyTorch time {t.elapsed}')

        with contexttimer.Timer() as t:
            for it in range(100):
                torch_result = self.torch_script_embedding(input_ids, token_type_ids, position_ids)
        print(f'TorchScript(i.e., jit) time {t.elapsed}')

        torch_result = self.torch_embedding(input_ids, token_type_ids, position_ids)
        ft_result = dlpack.from_dlpack(self.ft_embedding(_(input_ids), _(position_ids), _(token_type_ids)).to_dlpack())
        with contexttimer.Timer() as t:
            for it in range(100):
                ft_result = dlpack.from_dlpack(
                    self.ft_embedding(_(input_ids), _(position_ids), _(token_type_ids)).to_dlpack())
        self.assertTrue(torch.max(torch_result - ft_result) < 1e-5)
        print(f'FastTransform time {t.elapsed}')


if __name__ == '__main__':
    unittest.main()
