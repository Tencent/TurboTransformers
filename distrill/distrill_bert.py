from transformers import DistilBertTokenizer, DistilBertModel
from transformers import BertTokenizer, BertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# inputs = torch.randint(low=0,
#                        high=cfg.vocab_size - 1,
#                        size=(1, 10),
#                        dtype=torch.long,
#                        device=torch.device("cpu:0"))

## distrillation model
model = DistilBertModel.from_pretrained("distilbert-base-uncased",
                                        return_dict=True)

## bert model
bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)

cfg = model.config
print(cfg)
print(inputs)
outputs = model(**inputs)
bert_outputs = bert_model(**inputs)

print(model)
print(bert_model)

# print(bert_outputs - outputs)
#
# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states)
