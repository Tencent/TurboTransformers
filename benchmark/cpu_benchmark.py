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
"""
turbo-transformers Benchmark Utils

Usage:
    benchmark <model_name> --seq_len=<int> [--framework=<f>] [--batch_size=<int>] [-n <int>] [--num_threads=<int>]

Options:
    --framework=<f>      The framework to test in (torch, torch_jit, turbo-transformers,
                            onnxruntime-cpu, onnxruntime-mkldnn) [default: turbo-transformers].
    --seq_len=<int>      The sequence length.
    --batch_size=<int>   The batch size [default: 1].
    -n <int>             The iteration count [default: 10000].
    --num_threads=<int>  The thread count [default: 1].
"""

import json
import os

import docopt


def benchmark_turbo_transformers(model_name: str, seq_len: int,
                                 batch_size: int, n: int, num_threads: int):
    import torch
    import transformers
    import contexttimer
    import turbo_transformers
    import cProfile
    import benchmark_helper
    turbo_transformers.set_num_threads(num_threads)

    if model_name == "bert":
        cfg = transformers.BertConfig()
        model = transformers.BertModel(cfg)
        model.eval()
        model = turbo_transformers.BertModel.from_torch(model)
    elif model_name == "albert":
        cfg = transformers.AlbertConfig()
        model = transformers.AlbertModel(cfg)
        model.eval()
        model = turbo_transformers.AlbertModel.from_torch(model)
    elif model_name == "roberta":
        cfg = transformers.RobertaConfig()
        model = transformers.RobertaModel(cfg)
        model.eval()
        model = turbo_transformers.RobertaModel.from_torch(model)
    else:
        raise (f"benchmark does not support {model_name}")

    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long)
    benchmark_helper.run_model(lambda: model(input_ids), False, n, batch_size,
                               seq_len, "turbo", num_threads)


def benchmark_torch(model_name: str, seq_len: int, batch_size: int, n: int,
                    num_threads: int):
    import torch
    import transformers
    import contexttimer
    import benchmark_helper
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
    benchmark_helper.run_model(lambda: model(input_ids), False, n, batch_size,
                               seq_len, "torch", num_threads)


def benchmark_torch_jit(model_name: str, seq_len: int, batch_size: int, n: int,
                        num_threads: int):
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
            "n_threads": num_threads
        }))


def main():
    args = docopt.docopt(__doc__)
    kwargs = {
        'model_name': args['<model_name>'],
        'seq_len': int(args['--seq_len']),
        'batch_size': int(args['--batch_size']),
        'n': int(args['-n']),
        'num_threads': int(args['--num_threads'])
    }

    if args['--framework'] == 'turbo-transformers':
        benchmark_turbo_transformers(**kwargs)
    elif args['--framework'] == 'torch':
        benchmark_torch(**kwargs)
    elif args['--framework'] == 'torch_jit':
        benchmark_torch_jit(**kwargs)
    elif args['--framework'] == 'onnxruntime-cpu':
        import benchmark_helper
        benchmark_helper.onnxruntime_benchmark_creator('CPU')(**kwargs)
    elif args['--framework'] == 'onnxruntime-mkldnn':
        import benchmark_helper
        benchmark_helper.onnxruntime_benchmark_creator('MKL-DNN')(**kwargs)
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
