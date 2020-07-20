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


def run_model(model,
              use_cuda,
              num_iter,
              batch_size,
              seq_len,
              framework_name,
              thread_num=1):
    # warm up
    import torch
    import contexttimer
    import json
    model()
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with contexttimer.Timer() as t:
        for it in range(num_iter):
            model()

    if not use_cuda:
        qps = num_iter / t.elapsed
        time_consume = t.elapsed
    else:
        end.record()
        torch.cuda.synchronize()
        torch_elapsed = start.elapsed_time(end) / 1e3
        qps = num_iter / torch_elapsed
        time_consume = torch_elapsed
    print(
        json.dumps({
            "QPS": qps,
            "elapsed": time_consume,
            "n": num_iter,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "framework": framework_name,
            "thread_num": thread_num,
        }))


def run_random_model(model, use_cuda, num_iter, max_seq_len, min_seq_len,
                     framework_name, thread_num, cfg):
    import torch
    import contexttimer
    import json
    import random
    # no warm-up, its for fix-length input.
    test_device = torch.device('cuda:0') if use_cuda else torch.device('cpu:0')
    request_list = []
    # make sure all benchmarking runtimes are using the same random distribution.
    random.seed(0)
    for i in range(num_iter):
        generated_seq_len = random.randint(min_seq_len, max_seq_len)
        input_ids = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(1, generated_seq_len),
                                  dtype=torch.long,
                                  device=test_device)
        request_list.append(input_ids)

    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with contexttimer.Timer() as t:
        for request in request_list:
            model(request)

    if not use_cuda:
        qps = num_iter / t.elapsed
        time_consume = t.elapsed
    else:
        end.record()
        torch.cuda.synchronize()
        torch_elapsed = start.elapsed_time(end) / 1e3
        qps = num_iter / torch_elapsed
        time_consume = torch_elapsed
    print(
        json.dumps({
            "QPS": qps,
            "elapsed": time_consume,
            "n": num_iter,
            "max_seq_len": max_seq_len,
            "min_seq_len": min_seq_len,
            "framework": framework_name,
            "thread_num": thread_num,
        }))


def generate_onnx_model(model_name: str, filename: str, seq_len: int,
                        batch_size: int, backend: str):
    import transformers
    import torch
    import os

    test_device = torch.device('cuda:0') if backend == "GPU" else torch.device(
        'cpu:0')
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
    model.to(test_device)

    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    with open(filename, 'wb') as outf:
        torch.onnx.export(model=model,
                          args=(input_ids, ),
                          f=outf,
                          input_names=['input'],
                          output_names=['output'])
        # dynamic_axes = {'input':[0, 1], 'output':[0, 1]})
        # If not intended to make onnxruntime support variable batch size and sequence length, you can unset the parameter `dynamic_axes`.
        outf.flush()
    return cfg.vocab_size


def onnxruntime_benchmark_creator(backend: str):
    def _impl_(model_name: str,
               seq_len: int,
               batch_size: int,
               n: int,
               num_threads: int = 1):
        import multiprocessing
        import os
        temp_fn = "/tmp/temp_onnx.model"
        p = multiprocessing.Pool(1)
        vocab_size = p.apply(generate_onnx_model,
                             args=(model_name, temp_fn, seq_len, batch_size,
                                   backend))
        p.close()
        import contexttimer
        import onnxruntime.backend
        import onnx
        import numpy
        import json
        if not onnxruntime.backend.supports_device(backend):
            raise RuntimeError(
                f"onnxruntime does not support {backend}, recompile it!")

        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)

        model = onnx.load_model(f=temp_fn)
        model = onnxruntime.backend.prepare(
            model=model,
            device=backend,
            graph_optimization_level=onnxruntime.GraphOptimizationLevel.
            ORT_ENABLE_ALL)
        input_ids = numpy.random.randint(low=0,
                                         high=vocab_size - 1,
                                         size=(batch_size, seq_len),
                                         dtype=numpy.int64)

        # a torch model to check correctness
        import transformers
        import torch
        torch.set_grad_enabled(False)
        torch_model = transformers.BertModel.from_pretrained(
            "bert-base-uncased")
        torch_model.eval()
        torch_res = torch_model(torch.tensor(input_ids))
        onnx_res = model.run(inputs=[input_ids])
        assert (numpy.max(numpy.abs(torch_res[0].cpu().numpy() - onnx_res[0]))
                < 0.01)

        with contexttimer.Timer() as t:
            for _ in range(n):
                model.run(inputs=[input_ids])

        print(
            json.dumps({
                "QPS": n / t.elapsed,
                "elapsed": t.elapsed,
                "n": n,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "framework": f"onnx_rt_{backend}",
                "n_threads": num_threads
            }))

    return _impl_
