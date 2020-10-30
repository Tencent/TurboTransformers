# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# inputs = torch.randint(low=0,
#                        high=cfg.vocab_size - 1,
#                        size=(1, 10),
#                        dtype=torch.long,
#                        device=torch.device("cpu:0"))

## distrillation model
model = DistilBertModel.from_pretrained("distilbert-base-uncased",
                                        return_dict=True)

## bert model
bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)

cfg = model.config
print(cfg)
print(inputs)
outputs = model(**inputs)
bert_outputs = bert_model(**inputs)

print(model)
print(bert_model)

# print(bert_outputs - outputs)
#
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)
