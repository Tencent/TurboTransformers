import torch.utils.dlpack as dlpack
import fast_transformers
from transformers import BertTokenizer

def load_bert_token(file_name="bert-base-chinese"):
    return BertTokenizer.from_pretrained(file_name)

def convert2ft_tensor(t):
    return fast_transformers.Tensor.from_dlpack(dlpack.to_dlpack(t))
