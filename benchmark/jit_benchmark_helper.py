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
import json
import os

import docopt

__all__ = ['benchmark_torch_jit']


def benchmark_torch_jit(model_name: str, seq_len: int, batch_size: int, n: int,
                        enable_random: bool, max_seq_len: int,
                        min_seq_len: int, num_threads: int, use_gpu: bool,
                        enable_mem_opt: bool):
    import transformers
    import contexttimer
    import torch.jit
    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(False)
    if model_name == "bert":
        cfg = transformers.BertConfig()
        model = transformers.BertModel(cfg)
    elif model_name == "albert":
        cfg = transformers.AlbertConfig()
        model = transformers.AlbertModel(cfg)
    elif model_name == "roberta":
        cfg = transformers.RobertaConfig()
        model = transformers.RobertaModel(cfg)
    else:
        raise (f"benchmark does not support {model_name}")
    model.eval()
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long)

    model = torch.jit.trace(model, (input_ids, ))

    with torch.jit.optimized_execution(True):
        model(input_ids)
        with contexttimer.Timer() as t:
            for _ in range(n):
                model(input_ids)

    print(
        json.dumps({
            "QPS": n / t.elapsed,
            "elapsed": t.elapsed,
            "n": n,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "framework": "torch_jit",
            "n_threads": num_threads,
            "model_name": model_name
        }))
