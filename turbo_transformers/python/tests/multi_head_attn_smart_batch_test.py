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

import turbo_transformers

import unittest
import sys
import torch
import os
import numpy

from onmt.modules.multi_headed_attn import MultiHeadedAttention

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_decoder_multi_headed_attention.txt"


def create_test(query_seq_len_list,
                key_seq_len_list,
                attn_type,
                pre_layernorm,
                post_add_input,
                with_quantize_dynamic=False,
                set_layer_cache=False):
    class TestMultiHeadedAttentionSmartBatch(unittest.TestCase):
        def init_data(self, use_cuda):
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(4)
                turbo_transformers.set_num_threads(4)

            torch.set_grad_enabled(False)
            self.head_count = 16
            self.model_dim = 1024  #self.model_dim should % self.head_count = 0
            self.size_per_head = int(self.model_dim / self.head_count)

            self.query_seq_len_list = query_seq_len_list
            self.key_seq_len_list = key_seq_len_list
            # build the torch model
            self.model = MultiHeadedAttention(self.head_count, self.model_dim)
            self.model.eval()

            if use_cuda:
                self.model.to(self.test_device)

            # prepare torch input data
            self.Q_list = []
            for query_seq_len in query_seq_len_list:
                Q = torch.rand(
                    size=(
                        1,
                        query_seq_len,  #from_seq
                        self.model_dim),
                    dtype=torch.float32,
                    device=self.test_device)
                self.Q_list.append(Q)

            self.K_list = []
            self.V_list = []
            for key_seq_len in key_seq_len_list:
                K = torch.rand(
                    size=(
                        1,
                        key_seq_len,  #from_seq
                        self.model_dim),
                    dtype=torch.float32,
                    device=self.test_device)

                V = torch.rand(
                    size=(
                        1,
                        key_seq_len,  #to_seq
                        self.model_dim),
                    dtype=torch.float32,
                    device=self.test_device)
                self.K_list.append(K)
                self.V_list.append(V)

            # prepare turbo smart batch model
            self.turbo_smart_pad = turbo_transformers.MultiHeadedAttentionSmartBatch.from_onmt(
                self.model)

        def check_torch_and_turbo(self, use_cuda, num_iter=1):
            self.init_data(use_cuda)

            device = "GPU" if use_cuda else "CPU"
            info = f"\"({device}, {set_layer_cache}, {pre_layernorm}, {post_add_input}, {attn_type})\""

            # TODO(jiaruifang) test scenario where mask is not None.
            attention_mask = None
            layer_cache_torch = None

            res_list = []
            for Q, K, V in zip(self.Q_list, self.K_list, self.V_list):
                res, _ = self.model(
                    Q if attn_type == "self" else K,  #K,
                    Q if attn_type == "self" else V,  #V,
                    Q,
                    mask=attention_mask,
                    layer_cache=None,  #layer_cache_torch
                    attn_type=attn_type)
                res_list.append(res)

            # concat res_list together
            for i in range(len(res_list)):
                if i == 0:
                    concat_res = res_list[i]
                else:
                    concat_res = torch.cat((concat_res, res_list[i]), 1)

            self.assertTrue(
                concat_res.size()[1] == sum(self.query_seq_len_list))

            # concat K, Q, V together
            for i in range(len(self.query_seq_len_list)):
                if i == 0:
                    concat_Q = self.Q_list[i]
                    concat_K = self.K_list[i]
                    concat_V = self.V_list[i]
                else:
                    concat_Q = torch.cat((concat_Q, self.Q_list[i]), 1)
                    concat_K = torch.cat((concat_K, self.K_list[i]), 1)
                    concat_V = torch.cat((concat_V, self.V_list[i]), 1)

            self.assertTrue(concat_Q.size()[1] == sum(self.query_seq_len_list))
            self.assertTrue(concat_K.size()[1] == sum(self.key_seq_len_list))
            self.assertTrue(concat_V.size()[1] == sum(self.key_seq_len_list))
            self.assertTrue(attn_type == "self" or attn_type == "context")

            pad_res, _ = self.turbo_smart_pad(concat_K,
                                              concat_V,
                                              concat_Q,
                                              self.query_seq_len_list,
                                              self.key_seq_len_list,
                                              mask=attention_mask,
                                              layer_cache=None,
                                              attn_type=attn_type)

            diff = pad_res - concat_res
            # print(diff)
            print(torch.max(diff))
            self.assertTrue(
                numpy.allclose(pad_res.cpu(),
                               concat_res.cpu(),
                               atol=1e-3,
                               rtol=1e-3))

        def test_multi_headed_attention(self):
            # self.check_torch_and_turbo(use_cuda=False)
            if torch.cuda.is_available() and \
                    turbo_transformers.config.is_compiled_with_cuda():
                self.check_torch_and_turbo(use_cuda=True)

    globals(
    )[f"TestMultiHeadedAttentionSmartBatch_{attn_type}_{pre_layernorm}_{post_add_input}_{with_quantize_dynamic}_{set_layer_cache}"] = TestMultiHeadedAttentionSmartBatch


with open(fname, "w") as fh:
    fh.write(", torch, q_torch, turbo_transformers\n")

for set_layer_cache in [False]:
    for post_add_input in [False, True]:
        for pre_layernorm in [False, True]:
            query_seq_len_list = [9, 7, 13]
            key_seq_len_list = [10, 19, 20]
            for type in ["context", "self"]:
                create_test(query_seq_len_list,
                            key_seq_len_list,
                            type,
                            pre_layernorm,
                            post_add_input,
                            with_quantize_dynamic=False,
                            set_layer_cache=set_layer_cache)

if __name__ == '__main__':
    unittest.main()
