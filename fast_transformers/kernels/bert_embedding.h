#include "dlpack/dlpack.h"
#include "fast_transformers/utils/extern_c_guard.h"

BEGIN_EXTERN_C

typedef struct {
  DLTensor word_embeddings_;
  DLTensor position_embeddings_;
  DLTensor token_type_embeddings_;
  DLTensor layer_norm_weights_;
  DLTensor layer_norm_biases_;
} FT_BERT_EMBEDDING_LAYER;

extern DLManagedTensor* ft_bert_embedding(
    FT_BERT_EMBEDDING_LAYER self,
    DLTensor input_ids,
    DLTensor token_type_ids,
    DLTensor position_ids);

END_EXTERN_C
