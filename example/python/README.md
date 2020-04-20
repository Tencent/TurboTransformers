###  使用turbo_transformers编写加速代码
[English Version](./README_en.md)
因为turbo-transformer关于bert加速的C++部分改写只做到pooler层，后续需要用户根据自己的需求来定制后处理部分。

以下是使用turbo_transformers加速NLP任务的步骤

1. 首先我们需要准备一个使用huggingface训练好的bert模型（可以是huggingface的任一类的模型，例如BertPreTrainedModel、BertForSequenceClassification，这里以BertForSequenceClassification类为例）代码示例中的sequence classification model可以从百度云下载，将这个文件夹放在与example文件的同目录下即可，链接:https://pan.baidu.com/s/1WzMIQ2I3ncXb9aPLTJ7QNQ  密码:hj18
2. 写个新的类实现加速，这个类要实现`__init__`, `__call__`, `from_torch`, `from_pretrained`这四个函数，类的实现代码和说明可以参考bert_for_sequence_classification_example.py

```python
# 导入相关加速包
import turbo_transformers
from turbo_transformers import PoolingType
from turbo_transformers import ReturnType

# 导入要加速的model的类，这里以BertForSequenceClassificatio为例
from transformers.modeling_bert import BertModel as TorchBertModel
from transformers import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification as TorchBertForSequenceClassification
import os
import torch
from typing import Optional


class BertForSequenceClassification:  # 新建一个新的class用于加速
    def __init__(self, bertmodel, classifier):  # init函数实现（可以照抄不改）
        self.bert = bertmodel
        self.classifier = classifier

    def __call__(self,  # call函数实现（可以照抄不改）
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
        logits = self.classifier(pooler_output)  # 这里为classifier的输出，若用户要输出其他类型，可以在这之后自定义
        return logits

    @staticmethod
    def from_torch(model: TorchBertModel,  # from_torch函数实现
                   device: Optional[torch.device] = None):
        if device is not None and 'cuda' in device.type and torch.cuda.is_available(
        ):
            model.to(device)
        bertmodel = turbo_transformers.BertModelWithPooler.from_torch(
            model.bert)
        # 以上代码可以照抄，不改
        # 注意这里classifier是BertForSequenceClassification的类成员，如果用户如果自定义了其他成员，需要在这里进行相应的修改
        return BertForSequenceClassification(bertmodel, model.classifier)

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None):
        # 先使用from_pretrained函数加载自己训练的模型
        torch_model = TorchBertForSequenceClassification.from_pretrained(
            model_id_or_path)
        # 然后再使用加速模型的初始化函数，获取加速模型
        model = BertForSequenceClassification.from_torch(torch_model, device)
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

# 使用4个线程进行Bert推理
turbo_transformers.set_num_threads(4)

model_id = os.path.join(os.path.dirname(__file__),
                        'test-seq-classification-model')  # huggingface的模型的路径
tokenizer = BertTokenizer.from_pretrained(model_id)  # tokenizer初始化
turbo_model = turbo_transformers.BertForSequenceClassification.from_pretrained(
    model_id, torch.device('cpu:0'))  # 加速模型的初始化

# 模型加载后进行predict
input_ids = torch.tensor(
    tokenizer.encode('测试一下bert模型的性能和精度是不是符合要求?',
                     add_special_tokens=True)).unsqueeze(0)
torch_result = turbo_model(input_ids)
print(torch_result)
# tensor([[ 0.1451, -0.0373]], grad_fn=<AddmmBackward>)
```
