from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from transformers.modeling_bert import BertIntermediate as TorchBertIntermediate
from fast_transformers.layers.modeling_bert import BertEmbeddings
from .converts import convert2ft_tensor

__all__ = ['convert_embeddings_layer']


def convert_embeddings_layer(bert_embedding: TorchBertEmbeddings) -> BertEmbeddings:
    params = {k: convert2ft_tensor(v) for k, v in
              bert_embedding.named_parameters()}
    return BertEmbeddings(params['word_embeddings.weight'],
                          params['position_embeddings.weight'],
                          params['token_type_embeddings.weight'],
                          params['LayerNorm.weight'],
                          params['LayerNorm.bias'], bert_embedding.dropout.p)
