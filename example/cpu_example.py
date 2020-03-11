import torch
import transformers
import easy_transformers

# ʹÓ4¸öÌËÐasy_transformers
easy_transformers.set_num_threads(4)
# µ÷ransformersÌ¹©µÄ¤ѵwģÐ
model = transformers.BertModel.from_pretrained(
    "/data/pre_bert")
model.eval()
# ԤѵwģÐµÄä
cfg = model.config
batch_size = 2
seq_len = 128
#Ë»úÎ±¾ÐÁ
torch.manual_seed(1)
input_ids = torch.randint(low=0,
                            high=cfg.vocab_size - 1,
                            size=(batch_size, seq_len),
                            dtype=torch.long)

torch.set_grad_enabled(False)
torch_res = model(input_ids) # sequence_output, pooled_output, (hidden_states), (attentions)
print(torch_res[0][:,0,:])  # »ñncoderµõ½µĵÚ»¸ö´̬
# tensor([[-1.4238,  1.0980, -0.3257,  ...,  0.7149, -0.3883, -0.1134],
#        [-0.8828,  0.6565, -0.6298,  ...,  0.2776, -0.4459, -0.2346]])

# ¹¹½¨bert-encoderµÄ£Ð£¬Ê³öst·½ʽpoolingµĽá
# }Ö·½ʽÔÈģÐ£¬ÕÀֱ½ӴÓytorchģÐÔÈ
ft_model = easy_transformers.BertModel.from_torch(model)
# ´ÓļþÔÈ
# model = easy_transformers.BertModel.from_pretrained("bert-base-chinese")
res = ft_model(input_ids)
print(res)
# tensor([[-1.4292,  1.0934, -0.3270,  ...,  0.7212, -0.3893, -0.1172],
#         [-0.8878,  0.6571, -0.6331,  ...,  0.2759, -0.4496, -0.2375]])
