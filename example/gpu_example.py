import os
import numpy
import torch
import transformers
import contexttimer
import fast_transformers

torch.set_grad_enabled(False)
test_device = torch.device('cuda:0')
# load model from file, adapted to offline enviroment
model_id = os.path.join(os.path.dirname(__file__),
                         '../fast_transformers/python/tests/test-model')
model_torch = transformers.BertModel.from_pretrained(model_id)
model_torch.eval()
model_torch.to(test_device)
# the following two ways are the same
# 1. load model from checkpoint in file
# model_ft = fast_transformers.BertModel.from_pretrained(model_id, test_device)
# 2. load model from pytorch model
model_ft = fast_transformers.BertModel.from_torch(model_torch, test_device)
cfg = model_torch.config  # type: transformers.BertConfig

batch_size, seq_len = 10, 40 
input_ids = torch.randint(low=0,
                          high=cfg.vocab_size - 1,
                          size=(batch_size, seq_len),
                          dtype=torch.long,
                          device=test_device)

torch_result = model_torch(input_ids)
torch_result = (torch_result[0][:, 0]).cpu().numpy()
print(torch_result)

ft_result = model_ft(input_ids)
ft_result = ft_result.cpu().numpy()
print(ft_result)
