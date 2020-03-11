"""
easy-transformers Benchmark Utils

Usage:
    benchmark <model> --seq_len=<int> [--framework=<f>] [--batch_size=<int>] [-n <int>]

Options:
    --framework=<f>      The framework to test in (torch, torch_jit, easy-transformers,
                            onnxruntime-cpu, onnxruntime-mkldnn) [default: easy-transformers].
    --seq_len=<int>      The sequence length.
    --batch_size=<int>   The batch size [default: 1].
    -n <int>             The iteration count [default: 10000].
"""

import json
import os

import docopt


def benchmark_easy_transformers(model: str, seq_len: int, batch_size: int,
                                n: int):
    import torch
    import transformers
    import contexttimer
    import easy_transformers

    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return
    test_device = torch.device('cuda:0')

    model_dir = os.path.join(os.path.dirname(__file__),
                             '../easy_transformers/python/tests/test-model')
    model = transformers.BertModel.from_pretrained(
        model_dir)  # type: transformers.BertModel
    model.to(test_device)
    model.eval()
    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    model = easy_transformers.BertModel.from_torch(model)

    model(input_ids)

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    with contexttimer.Timer() as t:
        for _ in range(n):
            model(input_ids)

    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        event_elapsed = start.elapsed_time(end) / 1e3

    print(
        json.dumps({
            "QPS": n / event_elapsed,
            "elapsed": event_elapsed,
            "n": n,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "framework": "easy_transformers"
        }))


def benchmark_torch(model: str, seq_len: int, batch_size: int, n: int):
    import torch
    import transformers
    import contexttimer
    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return

    test_device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    model_dir = os.path.join(os.path.dirname(__file__),
                             '../easy_transformers/python/tests/test-model')
    model = transformers.BertModel.from_pretrained(
        model_dir)  # type: transformers.BertModel
    model.eval()

    model.to(test_device)

    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    model(input_ids)

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    with contexttimer.Timer() as t:
        for _ in range(n):
            model(input_ids)

    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        event_elapsed = start.elapsed_time(end) / 1e3

    print(
        json.dumps({
            "QPS": n / event_elapsed,
            "elapsed": event_elapsed,
            "n": n,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "framework": "torch"
        }))


def generate_onnx_model(model: str, filename: str, seq_len: int,
                        batch_size: int):
    import transformers
    import torch

    test_device = torch.device('cuda:0')

    torch.set_grad_enabled(False)
    model_dir = os.path.join(os.path.dirname(__file__),
                             '../easy_transformers/python/tests/test-model')
    model = transformers.BertModel.from_pretrained(
        model_dir)  # type: transformers.BertModel

    model.eval()
    model.to(test_device)

    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device = test_device)
    with open(filename, 'wb') as outf:
        torch.onnx.export(model=model, args=(input_ids, ), f=outf)
        outf.flush()
    return cfg.vocab_size


def onnxruntime_benchmark_creator(backend: str):
    def _impl_(model: str, seq_len: int, batch_size: int, n):
        import multiprocessing
        temp_fn = "/tmp/temp_onnx.model"
        p = multiprocessing.Pool(1)
        vocab_size = p.apply(generate_onnx_model,
                             args=(model, temp_fn, seq_len, batch_size))
        p.close()
        import contexttimer
        import os
        import onnx
        import onnxruntime  
        import onnxruntime.backend
        import time
        import numpy
        import torch
        model = onnx.load_model(f=temp_fn)
        test_device = torch.device('cuda:0')
        model = onnxruntime.backend.prepare(
            model=model,
            device=backend,
            graph_optimization_level=onnxruntime.GraphOptimizationLevel.
            ORT_ENABLE_ALL)
        input_ids = numpy.random.randint(low=0,
                                         high=vocab_size - 1,
                                         size=(batch_size, seq_len),
                                         dtype=numpy.int64
                                         )
        model.run(inputs=[input_ids])
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
            }))

    return _impl_

def main():

    args = docopt.docopt(__doc__)
    kwargs = {
        'model': args['<model>'],
        'seq_len': int(args['--seq_len']),
        'batch_size': int(args['--batch_size']),
        'n': int(args['-n']),
    }

    if args['--framework'] == 'easy-transformers':
        benchmark_easy_transformers(**kwargs)
    elif args['--framework'] == 'torch':
        benchmark_torch(**kwargs)
    elif args['--framework'] == 'onnxruntime':
        onnxruntime_benchmark_creator('GPU')(**kwargs)
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
