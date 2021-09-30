# Copyright (C) 2020 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

from .modeling_bert import BertEmbeddings, BertIntermediate, BertOutput, BertAttention, BertLayer, SequencePool, \
    BertEncoder, BertModel, PoolingType, BertPooler
from .qmodeling_bert import QBertIntermediate, QBertOutput, QBertLayer, QBertEncoder, QBertModel

from .modeling_albert import AlbertEmbeddings, AlbertAttention, AlbertLayer, AlbertTransformer, AlbertModel
from .modeling_decoder import MultiHeadedAttention, PositionwiseFeedForward, TransformerDecoderLayer, TransformerDecoder
from .modeling_encoder import TransformerEncoderLayer, TransformerEncoder
from .modeling_roberta import RobertaModel
from .modeling_gpt2 import GPT2Model
from .modeling_distillbert import DistillBertAttention, DistrillFFN, DistrillTransformerBlock, DistrillTransformer, DistilBertModel
from .modeling_smart_pad import MultiHeadedAttentionSmartBatch, BertLayerSmartBatch, BertEncoderSmartBatch, BertModelSmartBatch

from .return_type import ReturnType

from .bert_tensor_usage import get_bert_tensor_usage_record
from .static_allocator import greedy_by_size_offset_calculation

__all__ = [
    'BertEmbeddings', 'BertIntermediate', 'BertOutput', 'BertAttention',
    'BertLayer', 'BertEncoder', 'BertModel', 'ReturnType', 'BertPooler',
    'SequencePool', 'PoolingType', 'MultiHeadedAttention',
    'PositionwiseFeedForward', 'AlbertLayer', 'AlbertEmbeddings',
    'AlbertAttention', 'AlbertTransformer', 'AlbertModel',
    'PositionwiseFeedForward', 'TransformerDecoderLayer', 'TransformerDecoder',
    'TransformerEncoderLayer', 'TransformerEncoder',
    'RobertaModel', 'QBertIntermediate', 'QBertOutput', 'QBertLayer',
    'QBertEncoder', 'QBertModel', 'GPT2Model', 'DistillBertAttention',
    'DistrillFFN', 'DistrillTransformerBlock', 'DistrillTransformer',
    'DistilBertModel', 'MultiHeadedAttentionSmartBatch', 'BertLayerSmartBatch',
    'BertEncoderSmartBatch', 'BertModelSmartBatch', 'DistilBertModel'
]
