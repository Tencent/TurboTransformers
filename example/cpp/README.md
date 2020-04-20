# Use C++ interface to use TurboTransformers
1. prepare your model in format of *.npz
bash $WORKSPACE/tools/convert_huggingface_bert_pytorch_to_npz.py bert-base-uncased bert.npz
move bert.npz to $YOUR_BUILD_DIR/example/cpp/models
2. run our example
```
./bert_model_example
```
