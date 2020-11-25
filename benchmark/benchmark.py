"""
turbo-transformers Benchmark Utils

Usage:
    benchmark <model_name> [--seq_len=<int>] [--framework=<f>] [--batch_size=<int>] [-n <int>] [--enable-random] [--min_seq_len=<int>] [--max_seq_len=<int>] [--use_gpu] [--num_threads=<int>] [--enable_mem_opt]

Options:
    --framework=<f>      The framework to test in (torch, torch_jit, turbo-transformers,
                            onnxruntime-cpu, onnxruntime-mkldnn) [default: turbo-transformers].
    --seq_len=<int>      The sequence length [default: 10].
    --batch_size=<int>   The batch size [default: 1].
    -n <int>             The iteration count [default: 10000].
    --enable-random      Enable request cache.
    --min_seq_len=<int>  Minimal sequence length generated when enable random [default: 5]
    --max_seq_len=<int>  Maximal sequence length generated when enable random [default: 50]
    --use_gpu            Enable GPU.
    --num_threads=<int>  The number of CPU threads. [default: 4]
    --enable_mem_opt     Use model aware memory optimization for BERT.
"""

import docopt
from turbo_benchmark_helper import benchmark_turbo_transformers
from torch_benchmark_helper import benchmark_torch
from jit_benchmark_helper import benchmark_torch_jit
from onnx_benchmark_helper import onnxruntime_benchmark_creator


def main():
    args = docopt.docopt(__doc__)
    kwargs = {
        'model_name': args['<model_name>'],
        'seq_len': int(args['--seq_len']),
        'batch_size': int(args['--batch_size']),
        'n': int(args['-n']),
        'enable_random': True if args['--enable-random'] else False,
        'min_seq_len': int(args['--min_seq_len']),
        'max_seq_len': int(args['--max_seq_len']),
        'num_threads': int(args['--num_threads']),
        'use_gpu': True if args['--use_gpu'] else False,
        'enable_mem_opt': True if args['--enable_mem_opt'] else False,
    }
    if (kwargs['model_name'] not in ['bert'
                                     'albert']
            or args['--framework'] != 'turbo-transformers'):
        kwargs['enable_mem_opt'] = False
    if args['--framework'] == 'turbo-transformers':
        benchmark_turbo_transformers(**kwargs)
    elif args['--framework'] == 'torch':
        benchmark_torch(**kwargs)
    elif args['--framework'] == 'torch_jit':
        benchmark_torch_jit(**kwargs)
    elif args['--framework'] == 'onnxruntime-gpu':
        onnxruntime_benchmark_creator('GPU')(**kwargs)
    elif args['--framework'] == 'onnxruntime-cpu':
        onnxruntime_benchmark_creator('CPU')(**kwargs)
    else:
        raise RuntimeError(f"Not supportted framework {args['--framework']}")


if __name__ == '__main__':
    main()
