import torch
import transformers
import turbo_transformers
from turbo_transformers.layers.utils import convert2tt_tensor, try_convert, convert_returns_as_type, ReturnType
import time

model = transformers.BertModel.from_pretrained('bert-base-uncased')
model.eval()
cfg = model.config
torch.set_grad_enabled(False)

bertmodel = model
qbertmodel = turbo_transformers.QBertModel.from_torch(bertmodel)
torchqbertmodel = torch.quantization.quantize_dynamic(bertmodel)

lens = [10,20,40,60,80,100,200,300]
loops = 10

for l in lens:
    input_ids = torch.randint(low=0, high=cfg.vocab_size-1, size=(1, l), dtype=torch.long)
    print("seq length =", l)

    start = time.time()
    for i in range(loops):
        res = bertmodel(input_ids)
    end = time.time()
    print("torch fp32 model QPS =", loops/(end-start))

    start = time.time()
    for i in range(loops):
        res2 = qbertmodel(input_ids)
    end = time.time()
    print("turbo fp32+int8 model QPS =", loops/(end-start))

    start = time.time()
    for i in range(loops):
        res3 = torchqbertmodel(input_ids)
    end = time.time()
    print("torch int8 model QPS =", loops/(end-start))

print("max error against torch fp32 =", max(
    torch.max(torch.abs(res[0]-res2[0])), 
    torch.max(torch.abs(res[1]-res2[1]))))
print("max error against torch int8 =", max(
    torch.max(torch.abs(res3[0]-res2[0])), 
    torch.max(torch.abs(res3[1]-res2[1]))))
print("max error between torch int8 and torch fp32 =", max(
    torch.max(torch.abs(res3[0]-res[0])), 
    torch.max(torch.abs(res3[1]-res[1]))))
