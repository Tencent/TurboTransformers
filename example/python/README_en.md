###  Customized your own python inference code.
Because TurboTransformer's reimplemented embedding + BERT encoder + pooler, users may need to customize the post-processing part according to their own needs.
We take a classfication task as an example. It requires a Linear Layer after pooler.

1. First of all, we need to prepare a bert-classification model trained using huggingface (which can be any kind of model of huggingface, such as BertPreTrainedModel, BertForSequenceClassification, here taking BertForSequenceClassification as an example) The sequence classification model in the code example can be downloaded from Baidu Cloud. Place it in the same directory as the example file, link: https://pan.baidu.com/s/1WzMIQ2I3ncXb9aPLTJ7QNQ Password: hj18
2. Write a new class to replace the original huggingface's implementation. This class needs to implement the four functions `__init__`,` __call__`, `from_torch`,` from_pretrained`. The implementation code and description of the class can refer to bert_for_sequence_classification_example.py
