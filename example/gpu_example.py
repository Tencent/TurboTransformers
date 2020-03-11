import os
import numpy
import torch
import transformers
from transformers import BertTokenizer
import contexttimer
import easy_transformers

torch.set_grad_enabled(False)
test_device = torch.device('cuda:0')
# load model from file, adapted to offline enviroment
model_id = os.path.join(os.path.dirname(__file__),
                         '../easy_transformers/python/tests/test-model')
model_torch = transformers.BertModel.from_pretrained(model_id)
model_torch.eval()
model_torch.to(test_device)
# the following two ways are the same
# 1. load model from checkpoint in file
# model_ft = easy_transformers.BertModel.from_pretrained(model_id, test_device)
# 2. load model from pytorch model
model_ft = easy_transformers.BertModel.from_torch(model_torch, test_device)
cfg = model_torch.config  # type: transformers.BertConfig

batch_size, seq_len = 10, 40 
tokenizer = BertTokenizer.from_pretrained(model_id)
input_ids = tokenizer.encode('²âһÏbertģÐµÄÔܺ;«¶Èǲ»Ê·ûÇ?')
input_ids = torch.tensor([input_ids],
                         dtype=torch.long,
                         device=test_device)
torch_result = model_torch(input_ids)
torch_result = (torch_result[0][:, 0]).cpu().numpy()
# print(torch_result)

ft_result = model_ft(input_ids)
ft_result = ft_result.cpu().numpy()
print(numpy.max(numpy.abs(ft_result) - numpy.abs(torch_result)))
