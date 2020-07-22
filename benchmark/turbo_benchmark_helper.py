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

__all__ = ['benchmark_turbo_transformers']


def benchmark_turbo_transformers(model_name: str, seq_len: int,
                                 batch_size: int, n: int, enable_random: bool,
                                 max_seq_len: int, min_seq_len: int,
                                 num_threads: int, use_gpu: bool):
    import torch
    import transformers
    import contexttimer
    import turbo_transformers
    import benchmark_helper
    turbo_transformers.set_num_threads(num_threads)
    test_device = torch.device('cuda:0') if use_gpu else torch.device('cpu:0')
    cfg = None
    if model_name == "bert":
        cfg = transformers.BertConfig()
        model = transformers.BertModel(cfg)
        model.to(test_device)
        model.eval()
        model = turbo_transformers.BertModel.from_torch(model)
    elif model_name == "albert":
        cfg = transformers.AlbertConfig()
        model = transformers.AlbertModel(cfg)
        model.to(test_device)
        model.eval()
        model = turbo_transformers.AlbertModel.from_torch(model)
    elif model_name == "roberta":
        cfg = transformers.RobertaConfig()
        model = transformers.RobertaModel(cfg)
        model.to(test_device)
        model.eval()
        model = turbo_transformers.RobertaModel.from_torch(model)
    else:
        raise (f"benchmark does not support {model_name}")

    if enable_random:
        benchmark_helper.run_random_model(model, use_gpu, n, max_seq_len,
                                          min_seq_len, "turbo", 1, cfg)
    else:
        input_ids = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(batch_size, seq_len),
                                  dtype=torch.long,
                                  device=test_device)

        benchmark_helper.run_model(lambda: model(input_ids), True, n,
                                   batch_size, seq_len, "turbo")
