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
    benchmark <model> --seq_len=<int> [--framework=<f>] [--batch_size=<int>] [-n <int>]

Options:
    --framework=<f>      The framework to test in (torch, torch_jit, turbo-transformers,
                            onnxruntime-cpu, onnxruntime-mkldnn) [default: turbo-transformers].
    --seq_len=<int>      The sequence length.
    --batch_size=<int>   The batch size [default: 1].
    -n <int>             The iteration count [default: 10000].
"""

import json
import os

import docopt


def benchmark_turbo_transformers(model: str, seq_len: int, batch_size: int,
                                 n: int):
    import torch
    import transformers
    import contexttimer
    import turbo_transformers
    import benchmark_helper

    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return
    test_device = torch.device('cuda:0')

    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(
        model_id)  # type: transformers.BertModel
    model.to(test_device)
    model.eval()
    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    model = turbo_transformers.BertModel.from_torch(model)

    benchmark_helper.run_model(lambda: model(input_ids), True, n, batch_size,
                               seq_len, "turbo")


def benchmark_torch(model: str, seq_len: int, batch_size: int, n: int):
    import torch
    import transformers
    import contexttimer
    import benchmark_helper
    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return

    test_device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    model_id = "bert-base-uncased"
    model = transformers.BertModel.from_pretrained(
        model_id)  # type: transformers.BertModel
    model.eval()

    model.to(test_device)

    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    benchmark_helper.run_model(lambda: model(input_ids), True, n, batch_size,
                               seq_len, "torch")


def main():
    import benchmark_helper
    args = docopt.docopt(__doc__)
    kwargs = {
        'model': args['<model>'],
        'seq_len': int(args['--seq_len']),
        'batch_size': int(args['--batch_size']),
        'n': int(args['-n']),
    }

    if args['--framework'] == 'turbo-transformers':
        benchmark_turbo_transformers(**kwargs)
    elif args['--framework'] == 'torch':
        benchmark_torch(**kwargs)
    elif args['--framework'] == 'onnxruntime':
        benchmark_helper.onnxruntime_benchmark_creator('GPU')(**kwargs)
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
