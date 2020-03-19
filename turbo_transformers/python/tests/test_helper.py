# Copyright 2020 Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contexttimer
import torch
import torch.jit
import torch.onnx


def run_model(model, use_cuda, num_iter=50):
    # warm up
    model()
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with contexttimer.Timer() as t:
        for it in range(num_iter):
            result = model()

    if use_cuda:
        end.record()
        torch.cuda.synchronize()
        torch_elapsed = start.elapsed_time(end) / 1e3
        qps = num_iter / torch_elapsed
        time_consume = torch_elapsed / num_iter
    else:
        qps = num_iter / t.elapsed
        time_consume = t.elapsed / num_iter
    return result, qps, time_consume
