import torch
import transformers
import turbo_transformers
from turbo_transformers.layers.utils import convert2tt_tensor, try_convert, convert_returns_as_type, ReturnType
import time

model = transformers.BertModel.from_pretrained('bert-base-uncased')
model.eval()
torch.set_grad_enabled(False)

intermediate = torch.quantization.quantize_dynamic(model.encoder.layer[0].intermediate)
qintermediate = turbo_transformers.QBertIntermediate.from_torch(model.encoder.layer[0].intermediate)


input = torch.rand(10, 768)
loops = 10000


start = time.time()
for i in range(loops):
    res = intermediate(input)
end = time.time()
print("torch layer IPS =", loops/(end-start))

start = time.time()
for i in range(loops):
    res2 = qintermediate(input)
end = time.time()
print("our layer IPS =", loops/(end-start))

assert torch.max(torch.abs(res-res2)) < 1e-3
print("ok")
