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

try:
    # `turbo_transformers_cxxd` is the name on debug mode
    import turbo_transformers.turbo_transformers_cxxd as cxx
except ImportError:
    import turbo_transformers.turbo_transformers_cxx as cxx
from typing import Union, Optional, Sequence
import torch
from .return_type import convert_returns_as_type, ReturnType
from .utils import try_convert, convert2tt_tensor, to_param_dict_convert_tt, to_param_dict, create_empty_if_none, AnyTensor

from transformers import GPT2Model as TorchGPT2Model

import enum
import numpy as np
import os

__all__ = ['GPT2Model']


class GPT2Model:
    def __init__(self, model: TorchGPT2Model, backend="onnxrt"):
        # TODO type of GPT2Model_nopooler is (onnx and torch)
        self.backend = backend
        if backend == "onnxrt":
            self.onnxmodel = model
        elif backend == "turbo":
            raise NotImplementedError

    def __call__(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_tuple=None,
            **kwargs,
    ):
        if self.backend == "turbo":
            raise NotImplementedError
        elif self.backend == "onnxrt":
            # if attention_masks is None:
            #     attention_masks = np.ones(inputs.size(), dtype=np.int64)
            # else:
            #     attention_masks = attention_masks.cpu().numpy()
            # if token_type_ids is None:
            #     token_type_ids = np.zeros(inputs.size(), dtype=np.int64)
            # else:
            #     token_type_ids = token_type_ids.cpu().numpy()
            data = [input_ids.cpu().numpy()]
            outputs = self.onnxmodel.run(inputs=data)
            for idx, item in enumerate(outputs):
                outputs[idx] = torch.tensor(item, device=input_ids.device)
            return outputs

    @staticmethod
    def from_torch(model: TorchGPT2Model,
                   device: Optional[torch.device] = None,
                   backend: Optional[str] = "onnxrt"):
        """
        Args:
            model : a PyTorch GPT2 Model
            device : cpu or GPU
            backend : a string to indicates kernel providers
            Four options. [onnxrt, turbo]
        """
        use_gpu = False
        if device is None:
            device = model.device
        # may need to move to GPU explicitly
        if 'cuda' in device.type and torch.cuda.is_available():
            model.to(device)
            if backend is None:
                backend = "onnxrt"  # On GPU turbo is faster
            use_gpu = True
        else:
            if backend is None:
                backend = "onnxrt"  # On CPU onnxrt is faster

        if backend == "turbo":
            raise ("Not Implemented GPT2 on Turbo Backend")

        if backend == "onnxrt":
            import onnx
            import onnxruntime
            import onnxruntime.backend
            # TODO(jiaruifang) Figure out the meaning of GPT2
            enable_past_input = False

            num_layer = model.config.n_layer
            present_names = [f'present_{i}' for i in range(num_layer)]
            output_names = ["last_state"] + present_names

            input_names = ['input_ids']
            dynamic_axes = {
                'input_ids': {
                    0: 'batch_size',
                    1: 'seq_len'
                },
                #'token_type_ids' : {0: 'batch_size', 1: 'seq_len'},
                #'attention_mask' : {0: 'batch_size', 1: 'seq_len'},
                'last_state': {
                    0: 'batch_size',
                    1: 'seq_len'
                }
            }
            for name in present_names:
                dynamic_axes[name] = {1: 'batch_size', 3: 'seq_len'}

            inputs = {
                'input_ids':
                torch.randint(32, [2, 32], dtype=torch.long).to(device)
            }
            if enable_past_input:
                past_names = [f'past_{i}' for i in range(num_layer)]
                input_names = [
                    'input_ids'
                ] + past_names  #+ ['token_type_ids', 'attention_mask']
                dummy_past = [
                    torch.zeros(list(outputs[1][0].shape))
                    for _ in range(num_layer)
                ]
                for name in past_names:
                    dynamic_axes[name] = {1: 'batch_size', 3: 'seq_len'}
                export_inputs = (
                    inputs['input_ids'], tuple(dummy_past)
                )  #, inputs['token_type_ids'], inputs['attention_mask'])
            else:
                export_inputs = (inputs['input_ids'])
            output_dir = './gpt2_onnx'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            onnx_model_path = os.path.join(
                output_dir, 'gpt2_past{}.onnx'.format(int(enable_past_input)))

            torch.onnx.export(model,
                              args=export_inputs,
                              f=onnx_model_path,
                              input_names=input_names,
                              output_names=output_names,
                              dynamic_axes=dynamic_axes,
                              opset_version=11,
                              do_constant_folding=True,
                              verbose=False)
            onnx_model = onnx.load_model(f=onnx_model_path)
            onnx_model = onnxruntime.backend.prepare(
                model=onnx_model,
                device='GPU' if use_gpu else 'CPU',
                graph_optimization_level=onnxruntime.GraphOptimizationLevel.
                ORT_ENABLE_ALL)
            return GPT2Model(onnx_model, "onnxrt")

    @staticmethod
    def from_pretrained(model_id_or_path: str,
                        device: Optional[torch.device] = None,
                        backend: Optional[str] = None):
        torch_model = TorchGPT2Model.from_pretrained(model_id_or_path)
        model = GPT2Model.from_torch(torch_model, device, backend)
        model.config = torch_model.config
        model._torch_model = torch_model  # prevent destroy torch model.
        return model

    @staticmethod
    def from_npz(file_name: str, config,
                 device: Optional[torch.device] = None):
        raise NotImplementedError
