import torch
import transformers
import turbo_transformers

# use 4 threads for infernec
turbo_transformers.set_num_threads(4)
# load model
model_id = os.path.join(os.path.dirname(__file__),
                        '../turbo_transformers/python/tests/test-model')
model = transformers.BertModel.from_pretrained(model_id)

model.eval()
cfg = model.config
batch_size = 2
seq_len = 128
torch.manual_seed(1)
input_ids = torch.randint(low=0,
                          high=cfg.vocab_size - 1,
                          size=(batch_size, seq_len),
                          dtype=torch.long)

torch.set_grad_enabled(False)
torch_res = model(
    input_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
print(torch_res[0][:, 0, :])
# tensor([[-1.4238,  1.0980, -0.3257,  ...,  0.7149, -0.3883, -0.1134],
#        [-0.8828,  0.6565, -0.6298,  ...,  0.2776, -0.4459, -0.2346]])

# there are two methods to load pretrained model.
# 1, from a torch model, which has loaded a pretrained model
ft_model = turbo_transformers.BertModel.from_torch(model)
# 2. directly load from checkpoint (torch saved model)
# model = turbo_transformers.BertModel.from_pretrained(model_id)
res = ft_model(input_ids)
print(res)
