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
import torch
import transformers
import turbo_transformers
import enum
import time
import sys


def serial_bert_inference(torch_model, input_list):
    res_list = []
    for input_seq in input_list:
        res, _ = torch_model(input_seq)
        res_list.append(res)

    for i in range(len(res_list)):
        if i == 0:
            concat_res = res_list[i]
        else:
            concat_res = torch.cat((concat_res, res_list[i]), 1)
    return concat_res


def batch_bert_inference(turbo_model, input_list, query_seq_len_list):
    res, _ = turbo_model(input_list, query_seq_len_list)
    return res


def test_smart_batch(use_cuda: bool):
    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')
    cfg = transformers.BertConfig(attention_probs_dropout_prob=0.0,
                                  hidden_dropout_prob=0.0)
    torch_model = transformers.BertModel(cfg)

    # model_id = "bert-base-uncased"
    # torch_model = transformers.BertModel.from_pretrained(model_id)
    torch_model.eval()
    torch_model.to(test_device)
    torch.set_grad_enabled(False)

    cfg = torch_model.config
    # use 4 threads for computing
    if not use_cuda:
        turbo_transformers.set_num_threads(4)

    # Initialize a turbo BertModel with smart batching from torch model.
    turbo_model = turbo_transformers.BertModelSmartBatch.from_torch(
        torch_model)

    # a batch of queries with different lengths.
    query_seq_len_list = [18, 2, 3, 51]
    input_list = []

    # generate random inputs. Of course you can use real data.
    for query_seq_len in query_seq_len_list:
        input_seq = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(1, query_seq_len),
                                  dtype=torch.long,
                                  device=test_device)
        input_list.append(input_seq)

    # start inference
    s_res = serial_bert_inference(torch_model, input_list)
    b_res = batch_bert_inference(turbo_model, input_list, query_seq_len_list)
    print(torch.max(torch.abs(b_res - s_res)))
    assert (torch.max(torch.abs(b_res - s_res)) < 1e-2)

    start_time = time.time()
    for i in range(10):
        serial_bert_inference(torch_model, input_list)
    end_time = time.time()
    print("\ntorch time consum: {}".format(end_time - start_time))

    start_time = time.time()
    for i in range(10):
        batch_bert_inference(turbo_model, input_list, query_seq_len_list)
    end_time = time.time()
    print("\nturbo time consum: {}".format(end_time - start_time))


if __name__ == "__main__":
    if torch.cuda.is_available():
        test_smart_batch(True)
    test_smart_batch(False)
