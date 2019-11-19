from transformers.modeling_bert import BertEmbeddings as TorchBertEmbeddings
from fast_transformers.fast_transformers_cxx import BERTEmbedding
from .converts import convert2ft_tensor

__all__ = ['convert_embeddings_layer']


def convert_embeddings_layer(bert_embedding: TorchBertEmbeddings) -> BERTEmbedding:
    print(dir(bert_embedding.dropout))
    params = {k: convert2ft_tensor(v) for k, v in
              bert_embedding.named_parameters()}
    return BERTEmbedding(params['word_embeddings.weight'],
                         params['position_embeddings.weight'],
                         params['token_type_embeddings.weight'],
                         params['LayerNorm.weight'],
                         params['LayerNorm.bias'], 0.0)
