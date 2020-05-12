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

# import related package
import turbo_transformers
from turbo_transformers import PoolingType
from turbo_transformers import ReturnType

# import the class of the acceleration model. here is the example of BertForSequenceClassification.
from transformers.modeling_bert import BertModel as TorchBertModel
from transformers import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification as TorchBertForSequenceClassification
import os
import torch
from typing import Optional


class BertForSequenceClassification:  # create a new class for speeding up
    def __init__(
            self, bertmodel, classifier
    ):  # the realization of the init function（we can just copy it）
        self.bert = bertmodel
        self.classifier = classifier

    def __call__(
            self,  # the realization of the call function（we can just copy it）
            inputs,
            attention_masks=None,
            token_type_ids=None,
            position_ids=None,
            pooling_type=PoolingType.FIRST,
            hidden_cache=None,
            output=None,
            return_type=None):
        pooler_output, _, _ = self.bert(inputs,
                                     attention_masks,
                                     token_type_ids,
                                     position_ids,
                                     pooling_type,
                                     hidden_cache,
                                     return_type=ReturnType.TORCH)
        logits = self.classifier(
            pooler_output
        )  # It's the output of classifier, if User want to output the other type, he can define them after that.
        return logits

    @staticmethod
    def from_torch(
            model: TorchBertModel,  # from_torch函数实现
            device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        bertmodel = turbo_transformers.BertModelWithPooler.from_torch(
            model.bert)
        # We can copy the following code and do not change it
        # Notice: classifier is the class member of BertForSequenceClassification. If user define the other class member,
        # they need modify it here.
        return BertForSequenceClassification(bertmodel, model.classifier)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        # First, Use the function of from_pretrained to load the model you trained.
        torch_model = TorchBertForSequenceClassification.from_pretrained(
            model_id_or_path)
        # Then, Use the init function of the acceleration model to get it.
        model = BertForSequenceClassification.from_torch(torch_model, device)
        model._torch_model = torch_model  # prevent destroy torch model.
        return model


# use 4 threads for BERT inference
turbo_transformers.set_num_threads(4)

model_id = os.path.join(
    os.path.dirname(__file__),
    'test-seq-classification-model')  # the model of huggingface's path
tokenizer = BertTokenizer.from_pretrained(
    model_id)  # the initialization of tokenizer
turbo_model = turbo_transformers.BertForSequenceClassification.from_pretrained(
    model_id,
    torch.device('cpu:0'))  # the initialization of the acceleration model

# predict after loading the model
input_ids = torch.tensor(
    tokenizer.encode('测试一下bert模型的性能和精度是不是符合要求?',
                     add_special_tokens=True)).unsqueeze(0)
torch_result = turbo_model(input_ids)
print(torch_result)
# tensor([[ 0.1451, -0.0373]], grad_fn=<AddmmBackward>)
