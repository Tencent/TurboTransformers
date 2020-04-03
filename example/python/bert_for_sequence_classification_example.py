# Copyright 2020 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import turbo_transformers
from turbo_transformers import PoolingType
from turbo_transformers import ReturnType
from transformers.modeling_bert import BertModel as TorchBertModel
from transformers import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification as TorchBertForSequenceClassification
import os
import torch
from typing import Optional


class BertForSequenceClassification:
    def __init__(self, bertmodel, classifier):
        self.bert = bertmodel
        self.classifier = classifier

    def __call__(self,
                 inputs,
                 attention_masks=None,
                 token_type_ids=None,
                 position_ids=None,
                 pooling_type=PoolingType.FIRST,
                 hidden_cache=None,
                 output=None,
                 return_type=None):
        pooler_output, _ = self.bert(inputs,
                                     attention_masks,
                                     token_type_ids,
                                     position_ids,
                                     pooling_type,
                                     hidden_cache,
                                     return_type=ReturnType.TORCH)
        logits = self.classifier(pooler_output)
        return logits

    @staticmethod
    def from_torch(model: TorchBertModel,
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        bertmodel = turbo_transformers.BertModelWithPooler.from_torch(
            model.bert)
        return BertForSequenceClassification(bertmodel, model.classifier)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        torch_model = TorchBertForSequenceClassification.from_pretrained(
            model_id_or_path)
        model = BertForSequenceClassification.from_torch(torch_model, device)
        model._torch_model = torch_model  # prevent destroy torch model.
        return model


model_id = os.path.join(os.path.dirname(__file__),
                        'test-seq-classification-model')
tokenizer = BertTokenizer.from_pretrained(model_id)
turbo_model = turbo_transformers.BertForSequenceClassification.from_pretrained(
    model_id, torch.device('cpu:0'))
input_ids = torch.tensor(
    tokenizer.encode('测试一下bert模型的性能和精度是不是符合要求?',
                     add_special_tokens=True)).unsqueeze(0)
torch_result = turbo_model(input_ids)
print(torch_result)
# tensor([[ 0.1451, -0.0373]], grad_fn=<AddmmBackward>)
