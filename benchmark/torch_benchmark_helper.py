__all__ = ['benchmark_torch']


def benchmark_torch(model_name: str, seq_len: int, batch_size: int, n: int,
                    enable_random: bool, max_seq_len: int, min_seq_len: int,
                    num_threads: int, use_gpu: bool, enable_mem_opt: bool):
    import torch
    import transformers
    import benchmark_helper

    test_device = torch.device('cuda:0') if use_gpu else torch.device('cpu:0')
    torch.set_grad_enabled(False)
    torch.set_num_threads(num_threads)

    cfg = None
    if model_name == "bert":
        cfg = transformers.BertConfig()
        model = transformers.BertModel(cfg)
    elif model_name == "albert":
        cfg = transformers.AlbertConfig()
        model = transformers.AlbertModel(cfg)
    elif model_name == "roberta":
        cfg = transformers.RobertaConfig()
        model = transformers.RobertaModel(cfg)
    elif model_name == "distilbert":
        cfg = transformers.DistilBertConfig()
        model = transformers.DistilBertModel(cfg)
    else:
        raise (f"benchmark does not support {model_name}")
    model.eval()
    model.to(test_device)

    # cfg = model.config  # type: transformers.BertConfig
    if enable_random:
        benchmark_helper.run_variable_model(model, use_gpu, n, max_seq_len,
                                            min_seq_len, "torch", num_threads,
                                            cfg, enable_mem_opt, model_name)
    else:
        input_ids = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(batch_size, seq_len),
                                  dtype=torch.long,
                                  device=test_device)
        benchmark_helper.run_model(lambda: model(input_ids), use_gpu, n,
                                   batch_size, seq_len, "torch", num_threads,
                                   enable_mem_opt, model_name)
