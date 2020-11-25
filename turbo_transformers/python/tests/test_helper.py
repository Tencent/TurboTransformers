import contexttimer
import torch
import torch.jit
import torch.onnx

import cProfile
import cProfile, pstats, io
from pstats import SortKey


def run_model(model, use_cuda, num_iter=50, use_profile=False):
    # warm up
    model()
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    if use_profile:
        pr = cProfile.Profile()
        pr.enable()

    with contexttimer.Timer() as t:
        for it in range(num_iter):
            result = model()

    if use_profile:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

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


# for debug
def show_tensor(T, info):
    if T is None:
        print(info, " None")
        return
    T = torch.clone(T)
    print(info, T.size())
    print(T.flatten()[0:10])
    print(T.flatten()[-10:])
    print(torch.sum(T.flatten()))
