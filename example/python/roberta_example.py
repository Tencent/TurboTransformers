import unittest
import torch
from transformers.modeling_roberta import RobertaModel, RobertaConfig
import numpy
import turbo_transformers
import sys
import os


def test(use_cuda):
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    turbo_transformers.set_num_threads(4)

    test_device = torch.device('cuda:0') if use_cuda else \
        torch.device('cpu:0')

    cfg = RobertaConfig()
    torch_model = RobertaModel(cfg)
    torch_model.eval()

    if torch.cuda.is_available():
        torch_model.to(test_device)

    turbo_model = turbo_transformers.RobertaModel.from_torch(
        torch_model, test_device)

    input_ids = torch.randint(low=0,
                              high=cfg.vocab_size - 1,
                              size=(1, 10),
                              dtype=torch.long,
                              device=test_device)

    torch_result = torch_model(input_ids)
    torch_result_final = torch_result[0].cpu().numpy()

    turbo_result = turbo_model(input_ids)
    turbo_result_final = turbo_result[0].cpu().numpy()

    # See the differences
    # print(numpy.size(torch_result_final), numpy.size(turbo_result_final))
    print(torch_result_final - turbo_result_final)
    assert (numpy.allclose(torch_result_final,
                           turbo_result_final,
                           atol=1e-3,
                           rtol=1e-3))


if __name__ == "__main__":
    test(use_cuda=False)
