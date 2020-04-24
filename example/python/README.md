## How to use examples
### prepare your model
Modify the corresponding parts in the cpu(gpu)_example.py
1. I want to use the PyTorch huggingface/transformers.
```
model_id = "bert-base-uncased"
```
2. I want to use a PyTorch saved model.
```
model_id = "your_saved_model" directory
```
3. I want to use a Tensorflow checkpoint model
cd /workspace
python tools/convert_huggingface_bert_pytorch_to_npz.py bert-based-uncased bert_torch.npz
```
tt_model = turbo_transformers.BertModelWithPooler.from_npz(
    '/workspace/bert_torch.npz', cfg)
```
### run examples
```
python cpu_example.py
python gpu_example.py
```

### How to customized your post-processing layers after BERT encoder
[Chinese Version](./README.md)
Because TurboTransformer has accelerated embedding + BERT encoder + pooler, which are major hotspots.
Users may have to customize the not so time-consuming post-processing layers according to their own needs.
We take a classfication task as an example. It requires a Linear Layer after pooler.

1. First of all, we also need to prepare a bert-classification model trained using huggingface (which can be any kind of model of huggingface, such as BertPreTrainedModel, BertForSequenceClassification, here taking BertForSequenceClassification as an example) The sequence classification model in the code example can be downloaded from Baidu Cloud. Place it in the same directory as the example file, link: https://pan.baidu.com/s/1WzMIQ2I3ncXb9aPLTJ7QNQ Password: hj18
2. Write a new class to replace the original huggingface's implementation. This class needs to implement the four functions `__init__`,` __call__`, `from_torch`,` from_pretrained`. The implementation code and description of the class can refer to bert_for_sequence_classification_example.py
