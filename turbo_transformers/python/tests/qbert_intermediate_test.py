import torch
import transformers
import turbo_transformers
from turbo_transformers.layers.utils import convert2tt_tensor, try_convert, convert_returns_as_type, ReturnType
import time

cfg = transformers.BertConfig()
model = transformers.BertModel(cfg)
model.eval()
torch.set_grad_enabled(False)

intermediate = torch.quantization.quantize_dynamic(model.encoder.layer[0].intermediate)
qintermediate = turbo_transformers.QBertIntermediate.from_torch(model.encoder.layer[0].intermediate)


lens = [10,20,40,60,80,100,200,300]
loops = 1

for l in lens:
    input = torch.rand(1, l, 768)
    print("seq length =", l)

    start = time.time()
    for i in range(loops):
        res = intermediate(input)
    end = time.time()
    print("torch int8 layer QPS =", loops/(end-start))

    start = time.time()
    for i in range(loops):
        res2 = qintermediate(input)
    end = time.time()
    print("turbo int8 layer QPS =", loops/(end-start))

assert torch.max(torch.abs(res-res2)) < 1e-3