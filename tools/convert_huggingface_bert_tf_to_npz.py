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

try:
    from transformers import TFBertModel
except ImportError:
    print("please install tensorflow 2.0 by run `pip install tensorflow`")
import numpy as np
import sys


# User should define the map between tf model's layer name to tt model's layer name
def build_dic(num_layers):
    dic = {
        'tf_bert_model/bert/embeddings/word_embeddings/weight:0':
        'embeddings.word_embeddings.weight',
        'tf_bert_model/bert/embeddings/position_embeddings/embeddings:0':
        'embeddings.position_embeddings.weight',
        'tf_bert_model/bert/embeddings/token_type_embeddings/embeddings:0':
        'embeddings.token_type_embeddings.weight',
        'tf_bert_model/bert/embeddings/LayerNorm/gamma:0':
        'embeddings.LayerNorm.weight',
        'tf_bert_model/bert/embeddings/LayerNorm/beta:0':
        'embeddings.LayerNorm.bias',
        'tf_bert_model/bert/pooler/dense/kernel:0': 'pooler.dense.weight',
        'tf_bert_model/bert/pooler/dense/bias:0': 'pooler.dense.bias'
    }

    for i in range(num_layers):
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/self/query/kernel:0'] = f'encoder.layer.{i}.attention.self.query.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/self/query/bias:0'] = f'encoder.layer.{i}.attention.self.query.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/self/key/kernel:0'] = f'encoder.layer.{i}.attention.self.key.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/self/key/bias:0'] = f'encoder.layer.{i}.attention.self.key.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/self/value/kernel:0'] = f'encoder.layer.{i}.attention.self.value.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/self/value/bias:0'] = f'encoder.layer.{i}.attention.self.value.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/output/dense/kernel:0'] = f'encoder.layer.{i}.attention.output.dense.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/output/dense/bias:0'] = f'encoder.layer.{i}.attention.output.dense.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/output/LayerNorm/gamma:0'] = f'encoder.layer.{i}.attention.output.LayerNorm.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/attention/output/LayerNorm/beta:0'] = f'encoder.layer.{i}.attention.output.LayerNorm.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/intermediate/dense/kernel:0'] = f'encoder.layer.{i}.intermediate.dense.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/intermediate/dense/bias:0'] = f'encoder.layer.{i}.intermediate.dense.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/output/dense/kernel:0'] = f'encoder.layer.{i}.output.dense.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/output/dense/bias:0'] = f'encoder.layer.{i}.output.dense.bias'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/output/LayerNorm/gamma:0'] = f'encoder.layer.{i}.output.LayerNorm.weight'
        dic[f'tf_bert_model/bert/encoder/layer_._{i}/output/LayerNorm/beta:0'] = f'encoder.layer.{i}.output.LayerNorm.bias'
    return dic


def trans_layer_name_tf2turbo(dic, name):
    return dic[name]


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: \n"
            "    convert_huggingface_bert_tf_to_npz.py model_name output_file")
        exit(0)
    model = TFBertModel.from_pretrained(sys.argv[1])
    cfg = model.config
    dic = build_dic(cfg.num_hidden_layers)
    names = [v.name for v in model.trainable_variables]
    weights = np.array(model.get_weights())

    arrays = {}
    for i in range(len(names)):
        arrays[trans_layer_name_tf2turbo(dic, names[i])] = weights[i]

    q_weight_key = 'self.query.weight'
    k_weight_key = 'self.key.weight'
    v_weight_key = 'self.value.weight'

    q_bias_key = 'self.query.bias'
    k_bias_key = 'self.key.bias'
    v_bias_key = 'self.value.bias'

    numpy_dict = {}

    for k in arrays.keys():
        if k.endswith(q_weight_key):
            ret = []
            ret.append(arrays[k])
            ret.append(arrays[k[:-len(q_weight_key)] + k_weight_key])
            ret.append(arrays[k[:-len(q_weight_key)] + v_weight_key])
            v = np.concatenate(ret, axis=1)
            numpy_dict[k[:-len(q_weight_key)] +
                       "qkv.weight"] = np.ascontiguousarray(v)
        elif k.endswith(q_bias_key):
            ret = []
            ret.append(arrays[k])
            ret.append(arrays[k[:-len(q_bias_key)] + k_bias_key])
            ret.append(arrays[k[:-len(q_bias_key)] + v_bias_key])
            v = np.ascontiguousarray(np.concatenate(ret, axis=0))
            numpy_dict[k[:-len(q_bias_key)] + 'qkv.bias'] = v
        elif any((k.endswith(suffix) for suffix in (k_weight_key, v_weight_key,
                                                    k_bias_key, v_bias_key))):
            continue
        else:
            numpy_dict[k] = np.ascontiguousarray(arrays[k])

    np.savez_compressed(sys.argv[2], **numpy_dict)


if __name__ == '__main__':
    main()
