#include "bert_embedding.h"

BEGIN_EXTERN_C
extern DLManagedTensor* ft_bert_embedding(
    FT_BERT_EMBEDDING_LAYER self,
    DLTensor input_ids,
    DLTensor token_type_ids,
    DLTensor position_ids) {
  // implment
  return nullptr; 
}
END_EXTERN_C
