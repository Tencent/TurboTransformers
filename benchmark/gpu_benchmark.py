"""
Fast-Transformers Benchmark Utils

Usage:
    benchmark <model> --seq_len=<int> [--framework=<f>] [--batch_size=<int>] [-n <int>]

Options:
    --framework=<f>      The framework to test in (torch, torch_jit, fast-transformers,
                            onnxruntime-cpu, onnxruntime-mkldnn) [default: fast-transformers].
    --seq_len=<int>      The sequence length.
    --batch_size=<int>   The batch size [default: 1].
    -n <int>             The iteration count [default: 10000].
"""

import json
import os

import docopt


def benchmark_fast_transformers(model: str, seq_len: int, batch_size: int,
                                n: int):
    import torch
    import transformers
    import contexttimer
    import fast_transformers

    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return
    test_device = torch.device('cuda:0')

    model_dir = os.path.join(os.path.dirname(__file__),
                             '../fast_transformers/python/tests/test-model')
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
    model = fast_transformers.BertModel.from_torch(model)

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
            "framework": "fast_transformers"
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
                             '../fast_transformers/python/tests/test-model')
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


def main():

    args = docopt.docopt(__doc__)
    kwargs = {
        'model': args['<model>'],
        'seq_len': int(args['--seq_len']),
        'batch_size': int(args['--batch_size']),
        'n': int(args['-n']),
    }

    if args['--framework'] == 'fast-transformers':
        benchmark_fast_transformers(**kwargs)
    elif args['--framework'] == 'torch':
        benchmark_torch(**kwargs)
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
