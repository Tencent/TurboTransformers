"""
Fast-Transformers Benchmark Utils

Usage:
    benchmark <model> --seq_len=<int> [--framework=<f>] [--batch_size=<int>] [-n <int>] [--num_threads=<int>]

Options:
    --framework=<f>      The framework to test in (torch, torch_jit, fast-transformers,
                            onnxruntime-cpu, onnxruntime-mkldnn) [default: fast-transformers].
    --seq_len=<int>      The sequence length.
    --batch_size=<int>   The batch size [default: 1].
    -n <int>             The iteration count [default: 10000].
    --num_threads=<int>  The thread count [default: 1].
"""

import json

import docopt


def benchmark_fast_transformers(model: str, seq_len: int, batch_size: int,
                                n: int, num_threads: int):
    import torch
    import transformers
    import contexttimer
    import fast_transformers
    test_device = torch.device('cuda:0')

    fast_transformers.set_num_threads(num_threads)

    model = transformers.BertModel.from_pretrained(
        model)  # type: transformers.BertModel
    model.to(test_device)
    cfg = model.config  # type: transformers.BertConfig
    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(batch_size, seq_len),
                              dtype=torch.long,
                              device=test_device)
    model = fast_transformers.BertModel.from_torch(model)

    with fast_transformers.gperf_guard(
            f"ft_{batch_size}_{seq_len}_{num_threads}.gperf"):
        model(input_ids)

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
            "framework": "fast_transformers",
            "n_threads": num_threads
        }))


def benchmark_torch(model: str, seq_len: int, batch_size: int, n: int,
                    num_threads: int):
    import torch
    import transformers
    import contexttimer
    test_device = torch.device('cuda:0')

    torch.set_num_threads(num_threads)
    torch.set_grad_enabled(False)
    model = transformers.BertModel.from_pretrained(
        model)  # type: transformers.BertModel
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
            "framework": "torch",
            "n_threads": num_threads
        }))


def main():
    import torch
    if not torch.cuda.is_available():
        return

    args = docopt.docopt(__doc__)
    kwargs = {
        'model': args['<model>'],
        'seq_len': int(args['--seq_len']),
        'batch_size': int(args['--batch_size']),
        'n': int(args['-n']),
        'num_threads': int(args['--num_threads'])
    }

    if args['--framework'] == 'fast-transformers':
        benchmark_fast_transformers(**kwargs)
    elif args['--framework'] == 'torch':
        benchmark_torch(**kwargs)
    elif args['--framework'] == 'torch_jit':
        benchmark_torch_jit(**kwargs)
    elif args['--framework'] == 'onnxruntime-cpu':
        onnxruntime_benchmark_creator('CPU')(**kwargs)
    elif args['--framework'] == 'onnxruntime-mkldnn':
        onnxruntime_benchmark_creator('MKL-DNN')(**kwargs)
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
