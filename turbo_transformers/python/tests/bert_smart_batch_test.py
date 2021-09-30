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

import torch
from transformers.models.bert.modeling_bert import BertConfig, BertLayer, BertAttention, BertModel
from onmt.modules.multi_headed_attn import MultiHeadedAttention
import sys
import os

sys.path.append(os.path.dirname(__file__))
import test_helper

fname = "tt_bertlayer_SmartBatch.txt"


def create_test(query_seq_len_list):
    class TestBertSmartBatch(unittest.TestCase):
        def init_bertlayer_models(self, use_cuda: bool) -> None:
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)

            self.torch_model = BertLayer(self.cfg)
            self.torch_model.eval()
            if use_cuda:
                self.torch_model.to(self.test_device)

            self.hidden_size = self.cfg.hidden_size

            self.turbo_model = turbo_transformers.BertLayerSmartBatch.from_torch(
                self.torch_model)

        def init_bert_models(self, use_cuda: bool) -> None:
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)

            self.torch_model = BertModel(self.cfg)
            self.torch_model.eval()
            if use_cuda:
                self.torch_model.to(self.test_device)

            self.hidden_size = self.cfg.hidden_size

            self.turbo_model = turbo_transformers.BertModelSmartBatch.from_torch(
                self.torch_model)

        def init_attn_models(self, use_cuda: bool) -> None:
            self.test_device = torch.device('cuda:0') if use_cuda else \
                torch.device('cpu:0')
            if not use_cuda:
                torch.set_num_threads(1)

            torch.set_grad_enabled(False)
            self.cfg = BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)

            # torch model is from ONMT
            # self.torch_model = MultiHeadedAttention(self.cfg.num_attention_heads, self.cfg.hidden_size)
            self.torch_model = BertAttention(self.cfg)
            self.torch_model.eval()
            if use_cuda:
                self.torch_model.to(self.test_device)

            self.hidden_size = self.cfg.hidden_size

            # self.turbo_model = turbo_transformers.MultiHeadedAttentionSmartBatch.from_onmt(
            #     self.torch_model)
            self.turbo_model = turbo_transformers.MultiHeadedAttentionSmartBatch.from_torch(
                self.torch_model)

        def init_inputs(self):
            # prepare torch input data
            self.input_list = []
            for query_seq_len in query_seq_len_list:
                Q = torch.rand(
                    size=(
                        1,
                        query_seq_len,  #from_seq
                        self.hidden_size),
                    dtype=torch.float32,
                    device=self.test_device)
                self.input_list.append(Q)

            # concat Qs together
            for i in range(len(query_seq_len_list)):
                if i == 0:
                    self.concat_Q = self.input_list[i]
                else:
                    self.concat_Q = torch.cat(
                        (self.concat_Q, self.input_list[i]), 1)

            self.assertTrue(self.concat_Q.size()[1] == sum(query_seq_len_list))

        def init_inputs_seq(self):
            # prepare torch input data
            self.input_list = []
            for query_seq_len in query_seq_len_list:
                input_seq = torch.randint(low=0,
                                          high=self.cfg.vocab_size - 1,
                                          size=(1, query_seq_len),
                                          dtype=torch.long,
                                          device=self.test_device)
                self.input_list.append(input_seq)

            # self.assertTrue(self.concat_Q.size()[1] == sum(query_seq_len_list))

        def check_bert_attn(self, use_cuda):
            self.init_attn_models(use_cuda)
            self.init_inputs()

            num_iter = 2
            device = "GPU" if use_cuda else "CPU"

            res_list = []
            for Q in self.input_list:
                # res, _ = self.torch_model(
                #     Q,
                #     Q,
                #     Q,
                #     mask=None,
                #     layer_cache=None,  #layer_cache_torch
                #     attn_type="self")
                # res_list.append(res)
                attention_mask = torch.ones((1, Q.size(1)),
                                            dtype=torch.float32,
                                            device=self.test_device)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * -10000.0
                res = self.torch_model(Q, attention_mask=None)
                res_list.append(res[0])

            # concat res_list together
            for i in range(len(res_list)):
                if i == 0:
                    concat_res = res_list[i]
                else:
                    concat_res = torch.cat((concat_res, res_list[i]), 1)

            pad_result, _ = self.turbo_model(self.concat_Q,
                                             self.concat_Q,
                                             self.concat_Q,
                                             query_seq_len_list, [],
                                             mask=None,
                                             layer_cache=None,
                                             post_layernorm=True,
                                             attn_type="self")

            # Tensor core will introduce more errors
            tolerate_error = 1e-2 if use_cuda else 1e-3
            self.assertTrue(
                torch.max(torch.abs(concat_res - pad_result)) < tolerate_error)

        def check_bert_layer(self, use_cuda):
            self.init_bertlayer_models(use_cuda)
            self.init_inputs()

            num_iter = 2
            device = "GPU" if use_cuda else "CPU"

            res_list = []
            for Q in self.input_list:
                res, _ = self.torch_model(Q, None, output_attentions=True)
                res_list.append(res)

            # concat res_list together
            for i in range(len(res_list)):
                if i == 0:
                    concat_res = res_list[i]
                else:
                    concat_res = torch.cat((concat_res, res_list[i]), 1)

            pad_result, _ = self.turbo_model(self.concat_Q,
                                             query_seq_len_list,
                                             attention_mask=None,
                                             output_attentions=False)

            # Tensor core will introduce more errors
            tolerate_error = 1e-2 if use_cuda else 1e-3
            self.assertTrue(
                torch.max(torch.abs(concat_res - pad_result)) < tolerate_error)

            # self.assertTrue(
            #     torch.max(
            #         torch.abs(torch_bert_layer_result[1] -
            #                   turbo_bert_layer_result[1])) < tolerate_error)

            # with open(fname, "a") as fh:
            #     fh.write(
            #         f"\"({batch_size},{seq_length:03})\", {torch_qps}, {turbo_qps}\n"
            #     )

        def check_bert_model(self, use_cuda):
            self.init_bert_models(use_cuda)
            self.init_inputs_seq()

            num_iter = 2
            device = "GPU" if use_cuda else "CPU"

            # for reference
            res_list = []
            for Q in self.input_list:
                res = self.torch_model(Q)
                res_list.append(res['last_hidden_state'])

            for i in range(len(res_list)):
                if i == 0:
                    concat_res = res_list[i]
                else:
                    concat_res = torch.cat((concat_res, res_list[i]), 1)

            # turbo inference
            pad_result, _ = self.turbo_model(self.input_list,
                                             query_seq_len_list)

            # Tensor core will introduce more errors
            tolerate_error = 1e-2 if use_cuda else 1e-3
            self.assertTrue(
                torch.max(torch.abs(concat_res - pad_result)) < tolerate_error)


        def test_bert(self):
            self.check_bert_model(use_cuda=False)
            self.check_bert_layer(use_cuda=False)
            self.check_bert_attn(use_cuda=False)
            if torch.cuda.is_available() and \
                    turbo_transformers.config.is_compiled_with_cuda():
                self.check_bert_model(use_cuda=True)
                self.check_bert_layer(use_cuda=True)
                self.check_bert_attn(use_cuda=True)

    globals()[f"TestBertSmartBatch"] = TestBertSmartBatch


with open(fname, "w") as fh:
    fh.write(", torch, turbo_transformers\n")

query_seq_list = [4, 1]
create_test(query_seq_list)

if __name__ == '__main__':
    unittest.main()
