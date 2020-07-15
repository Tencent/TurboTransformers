import torch
import transformers
from  turbo_transformers.layers.utils import convert2tt_tensor, try_convert, convert_returns_as_type, ReturnType
import os
import sys

class QBertIntermediate():
    def __init__(self, dense, act):
        self.qlinear = dense
        self.act = act # Replace it with turbo API
    def __call__(self, input_tensor):
        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = convert_returns_as_type(input_tensor, ReturnType.TORCH)
        return self.act(self.qlinear(input_tensor))
    @staticmethod
    def from_torch(intermediate):
        return QBertIntermediate(intermediate.dense, intermediate.intermediate_act_fn)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

model = transformers.BertModel.from_pretrained('bert-base-uncased')
model = torch.quantization.quantize_dynamic(model)
model.eval()

intermediate = model.encoder.layer[0].intermediate
qintermediate = QBertIntermediate.from_torch(intermediate)
# qlinear = intermediate.dense
# print_size_of_model(qlinear)
# packed_wb = qlinear._packed_params._packed_params
# print("packed_wb (Byte) :", sys.getsizeof(packed_wb.storage()))
# wb = torch.ops.quantized.linear_unpack(packed_wb)
# print("unpacked w (Byte) :", sys.getsizeof(wb[0].storage()))

input = torch.rand(10, 768)
res = intermediate(input)
res2 = qintermediate(input)
# res2 = convert_returns_as_type(res2, ReturnType.TORCH)
assert res.equal(res2)
print("ok")
