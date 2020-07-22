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

enable_latency_plot = 1


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
    test_device = torch.device('cuda:0') if use_cuda else torch.device('cpu:0')
    # warm-up using the longest sequence
    # TODO(jiaruifang) We know recommend you to run warm-up before inference.
    # In the future we will refactor allocator so as to not avoid warm-up
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(1, max_seq_len),
                              dtype=torch.long,
                              device=test_device)
    model(input_ids)
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
    if enable_latency_plot:
        # warm-up
        import time
        print(f"dump results to {framework_name}_latency.txt")
        with open(f"{framework_name}_latency.txt", "w") as of:
            result_list = []
            model(
                torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(1, max_seq_len),
                              dtype=torch.long,
                              device=test_device))
            for request in request_list:
                if use_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                with contexttimer.Timer() as t:
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
                result_list.append([len(request.view(-1)), time_consume])
            elapse = 0.
            result_list = sorted(result_list, key=lambda s: s[0])
            for item in result_list:
                of.write(f"{item[0]}, {item[1]}\n")
                elapse += item[1]
            print(f"elapsed {elapse}  QPS {num_iter/elapse}")
    else:
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


def generate_onnx_model(model_name: str,
                        filename: str,
                        seq_len: int,
                        batch_size: int,
                        backend: str,
                        use_dynamic_axes: bool = False):
    import transformers
    import torch
    import os

    test_device = torch.device('cuda:0') if backend == "GPU" else torch.device(
        'cpu:0')
    torch.set_grad_enabled(False)

    if model_name == "bert":
        # use a real model to check the correctness
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
        # cfg = transformers.BertConfig()
        # model = transformers.BertModel(cfg)
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
        if not use_dynamic_axes:
            torch.onnx.export(model=model, args=(input_ids, ), f=outf)
        else:
            torch.onnx.export(model=model,
                              args=(input_ids, ),
                              f=outf,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={
                                  'input': [0, 1],
                                  'output': [0, 1]
                              },
                              opset_version=12)
        # If not intended to make onnxruntime support variable batch size and sequence length,
        # you can unset the parameter `dynamic_axes`.
        outf.flush()
    return cfg.vocab_size, cfg


def onnxruntime_benchmark_creator(backend: str):
    def _impl_(model_name: str,
               seq_len: int,
               batch_size: int,
               n: int,
               enable_random: bool,
               max_seq_len: int,
               min_seq_len: int,
               num_threads: int = 1):
        use_cuda = True
        import multiprocessing
        import os
        temp_fn = "/tmp/temp_onnx.model"
        p = multiprocessing.Pool(1)
        vocab_size, cfg = p.apply(generate_onnx_model,
                                  args=(model_name, temp_fn, seq_len,
                                        batch_size, backend, enable_random))
        p.close()
        import contexttimer
        import onnxruntime.backend
        import onnx
        import numpy
        import json
        import random

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
        # Prepare a torch bert model to check correctness if benchmarking bert
        if model_name == "bert":
            import transformers
            import torch
            torch.set_grad_enabled(False)
            torch_model = transformers.BertModel.from_pretrained(
                "bert-base-uncased")

            if enable_random:
                input_ids = numpy.random.randint(low=0,
                                                 high=cfg.vocab_size - 1,
                                                 size=(2, 17),
                                                 dtype=numpy.int64)
            else:
                input_ids = numpy.random.randint(low=0,
                                                 high=cfg.vocab_size - 1,
                                                 size=(batch_size, seq_len),
                                                 dtype=numpy.int64)
            torch_model.eval()
            torch_res = torch_model(torch.tensor(input_ids))
            onnx_res = model.run(inputs=[input_ids])
            assert (numpy.max(
                numpy.abs(torch_res[0].cpu().numpy() - onnx_res[0])) < 0.01)

        if enable_random:
            request_list = []
            random.seed(0)
            for i in range(n):
                generated_seq_len = random.randint(min_seq_len, max_seq_len)
                input_ids = numpy.random.randint(low=0,
                                                 high=cfg.vocab_size - 1,
                                                 size=(1, generated_seq_len),
                                                 dtype=numpy.int64)
                request_list.append(input_ids)

            if enable_latency_plot:
                import time
                print(f"dump results to onnxrt_latency.txt")
                result_list = []
                with open(f"onnxrt_latency.txt", "w") as of:
                    for request in request_list:
                        if use_cuda:
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()

                        with contexttimer.Timer() as t:
                            model.run(inputs=[request])

                        if not use_cuda:
                            qps = n / t.elapsed
                            time_consume = t.elapsed
                        else:
                            end.record()
                            torch.cuda.synchronize()
                            torch_elapsed = start.elapsed_time(end) / 1e3
                            qps = n / torch_elapsed
                            time_consume = torch_elapsed
                        result_list.append(
                            [len(request.flatten()), time_consume])
                    elapse = 0.
                    result_list = sorted(result_list, key=lambda s: s[0])
                    for item in result_list:
                        of.write(f"{item[0]}, {item[1]}\n")
                        elapse += item[1]
                    print(f"elapsed {elapse} QPS {n/elapse}")
            else:
                if use_cuda:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                with contexttimer.Timer() as t:
                    for request in request_list:
                        model.run(inputs=[request])

                if not use_cuda:
                    qps = n / t.elapsed
                    time_consume = t.elapsed
                else:
                    end.record()
                    torch.cuda.synchronize()
                    torch_elapsed = start.elapsed_time(end) / 1e3
                    qps = n / torch_elapsed
                    time_consume = torch_elapsed
        else:
            input_ids = numpy.random.randint(low=0,
                                             high=vocab_size - 1,
                                             size=(batch_size, seq_len),
                                             dtype=numpy.int64)
            with contexttimer.Timer() as t:
                for _ in range(n):
                    model.run(inputs=[input_ids])

            if enable_random:
                print(
                    json.dumps({
                        "QPS": qps,
                        "elapsed": time_consume,
                        "n": n,
                        "max_seq_len": max_seq_len,
                        "min_seq_len": min_seq_len,
                        "framework": f"onnx_rt_{backend}",
                        "thread_num": num_threads,
                    }))
            else:
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
