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

from transformers import BertConfig
try:
    import tensorflow as tf
except ImportError:
    print("please install tensorflow 2.0 by run `pip install tensorflow`")
import numpy as np
import sys
import os


# User should define the map between tf model's layer name to tt model's layer name
def build_dic(num_layers):
    dic = {
        'bert/embeddings/word_embeddings':
        'embeddings.word_embeddings.weight',
        'bert/embeddings/position_embeddings':
        'embeddings.position_embeddings.weight',
        'bert/embeddings/token_type_embeddings':
        'embeddings.token_type_embeddings.weight',
        'bert/embeddings/LayerNorm/gamma':
        'embeddings.LayerNorm.weight',
        'bert/embeddings/LayerNorm/beta':
        'embeddings.LayerNorm.bias',
        'bert/pooler/dense/kernel': 'pooler.dense.weight',
        'bert/pooler/dense/bias': 'pooler.dense.bias'
    }

    for i in range(num_layers):
        dic[f'bert/encoder/layer_{i}/attention/self/query/kernel'] = f'encoder.layer.{i}.attention.self.query.weight'
        dic[f'bert/encoder/layer_{i}/attention/self/query/bias'] = f'encoder.layer.{i}.attention.self.query.bias'
        dic[f'bert/encoder/layer_{i}/attention/self/key/kernel'] = f'encoder.layer.{i}.attention.self.key.weight'
        dic[f'bert/encoder/layer_{i}/attention/self/key/bias'] = f'encoder.layer.{i}.attention.self.key.bias'
        dic[f'bert/encoder/layer_{i}/attention/self/value/kernel'] = f'encoder.layer.{i}.attention.self.value.weight'
        dic[f'bert/encoder/layer_{i}/attention/self/value/bias'] = f'encoder.layer.{i}.attention.self.value.bias'
        dic[f'bert/encoder/layer_{i}/attention/output/dense/kernel'] = f'encoder.layer.{i}.attention.output.dense.weight'
        dic[f'bert/encoder/layer_{i}/attention/output/dense/bias'] = f'encoder.layer.{i}.attention.output.dense.bias'
        dic[f'bert/encoder/layer_{i}/attention/output/LayerNorm/gamma'] = f'encoder.layer.{i}.attention.output.LayerNorm.weight'
        dic[f'bert/encoder/layer_{i}/attention/output/LayerNorm/beta'] = f'encoder.layer.{i}.attention.output.LayerNorm.bias'
        dic[f'bert/encoder/layer_{i}/intermediate/dense/kernel'] = f'encoder.layer.{i}.intermediate.dense.weight'
        dic[f'bert/encoder/layer_{i}/intermediate/dense/bias'] = f'encoder.layer.{i}.intermediate.dense.bias'
        dic[f'bert/encoder/layer_{i}/output/dense/kernel'] = f'encoder.layer.{i}.output.dense.weight'
        dic[f'bert/encoder/layer_{i}/output/dense/bias'] = f'encoder.layer.{i}.output.dense.bias'
        dic[f'bert/encoder/layer_{i}/output/LayerNorm/gamma'] = f'encoder.layer.{i}.output.LayerNorm.weight'
        dic[f'bert/encoder/layer_{i}/output/LayerNorm/beta'] = f'encoder.layer.{i}.output.LayerNorm.bias'
    return dic


def trans_layer_name_tf2turbo(dic, name):
    return dic[name]


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: \n"
            "    convert_tf_bert_to_npz.py model_name output_file")
        exit(0)
    model_path = sys.argv[1]
    ckpt_path = os.path.join(model_path, "bert_model.ckpt")
    cfg = BertConfig.from_pretrained(os.path.join(model_path, "bert_config.json"))
    dic = build_dic(cfg.num_hidden_layers)
    names = [v[0] for v in tf.train.list_variables(ckpt_path)]

    arrays = {}
    for i in range(len(names)):
        if names[i].startswith("cls"):
            continue
        arrays[trans_layer_name_tf2turbo(dic, names[i])] = tf.train.load_variable(ckpt_path, names[i])

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
