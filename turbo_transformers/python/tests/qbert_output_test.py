import torch
import transformers
import turbo_transformers
from turbo_transformers.layers.utils import convert2tt_tensor, try_convert, convert_returns_as_type, ReturnType
import time

cfg = transformers.BertConfig()
model = transformers.BertModel(cfg)
model.eval()
torch.set_grad_enabled(False)

bertoutput = torch.quantization.quantize_dynamic(model.encoder.layer[0].output)
qbertoutput = turbo_transformers.QBertOutput.from_torch(model.encoder.layer[0].output)


lens = [10,20,40,60,80,100,200,300]
loops = 1

for l in lens:
    hidden = torch.rand(1, l, 3072)
    input = torch.rand(1, l, 768)
    print("seq length =", l)

    start = time.time()
    for i in range(loops):
        res = bertoutput(hidden, input)
    end = time.time()
    print("torch int8 layer QPS =", loops/(end-start))

    start = time.time()
    for i in range(loops):
        res2 = qbertoutput(hidden, input)
    end = time.time()
    print("turbo int8 layer QPS =", loops/(end-start))

assert torch.max(torch.abs(res-res2)) < 1e-3
