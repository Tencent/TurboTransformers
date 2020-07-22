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
              use_gpu,
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
    if use_gpu:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with contexttimer.Timer() as t:
        for it in range(num_iter):
            model()

    if not use_gpu:
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


def run_variable_model(model, use_gpu, num_iter, max_seq_len, min_seq_len,
                       framework_name, thread_num, cfg):
    import torch
    import contexttimer
    import json
    import random
    test_device = torch.device('cuda:0') if use_gpu else torch.device('cpu:0')
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
            # model(
            #     torch.randint(low=0,
            #                   high=cfg.vocab_size - 1,
            #                   size=(1, max_seq_len),
            #                   dtype=torch.long,
            #                   device=test_device))
            for request in request_list:
                model(request)  #TODO warmup
                if use_gpu:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                with contexttimer.Timer() as t:
                    model(request)

                if not use_gpu:
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
        if use_gpu:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        with contexttimer.Timer() as t:
            for request in request_list:
                model(request)

        if not use_gpu:
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
