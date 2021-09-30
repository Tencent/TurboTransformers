## How to use examples
### prepare your model
Modify the corresponding parts in the cpu(gpu)_example.py
1. I want to use the PyTorch huggingface/transformers.
```
model_id = "bert-base-uncased"
```
2. I want to use a PyTorch saved model

We can load a model from the directory of pre-trained model.
```
model_id = "your_saved_model" directory
```

3. I want to use a Tensorflow checkpoint model

```
cd /workspace
python tools/convert_huggingface_bert_tf_to_npz.py bert-base-uncased /workspace/bert_tf.npz
```
update the corresponding line in bert_example.py
```
tt_model = turbo_transformers.BertModel.from_npz(
    '/workspace/bert_tf.npz', cfg)
```
### run examples
```
python bert_example.py
```

**Attention** : If you want to use turbo with C++ backend instead of onnxrt.
Directly linking an MKL of Pytorch installed by conda will lead to poor performance
in our hand-crafted C++ version.
You should install an official MKL an set MKL PATH in CMakeLists.txt.
As a not so elegant alternative, you can uninstall OpenNMT-py and downgrade torch to 1.1.0.

I have prepared an image for bert only runtime on dockerhub with .

`thufeifeibear/turbo_transformers_cpu:bert_only_v0.1`

**Attention** : If you want to use turbo with C++ backend instead of onnxrt.
Directly linking an MKL of Pytorch installed by conda will lead to poor performance
in our hand-crafted C++ version.
You should install an official MKL an set MKL PATH in CMakeLists.txt.
As a not so elegant alternative, you can uninstall OpenNMT-py and downgrade torch to 1.1.0.

I have prepared an image for bert only runtime on dockerhub with .

`thufeifeibear/turbo_transformers_cpu:bert_only_v0.1`

### How to customized your post-processing layers after BERT encoder
[Chinese Version](./README.md)
Because TurboTransformer has accelerated embedding + BERT encoder + pooler, which are major hotspots.
Users may have to customize the not so time-consuming post-processing layers according to their own needs.
We take a classfication task as an example. It requires a Linear Layer after pooler.

1. First of all, we also need to prepare a bert-classification model trained using huggingface (which can be any kind of model of huggingface, such as BertPreTrainedModel, BertForSequenceClassification, here taking BertForSequenceClassification as an example) The sequence classification model in the code example can be downloaded from Baidu Cloud. Place it in the same directory as the example file, link: https://pan.baidu.com/s/1WzMIQ2I3ncXb9aPLTJ7QNQ Password: hj18
2. Write a new class to replace the original huggingface's implementation. This class needs to implement the four functions `__init__`,` __call__`, `from_torch`,` from_pretrained`. The implementation code and description of the class can refer to bert_for_sequence_classification_example.py
