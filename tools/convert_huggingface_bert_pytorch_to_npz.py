from transformers.modeling_bert import BertModel
import sys
import numpy
import torch

# Attention: weight.dense of QKV, intermediate, output should be stored as (:, hidden_dim)
# While pytorch store them as (hidden_dim, :)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: \n"
            "    convert_huggingface_bert_to_npz model_name (bert-base-uncased) output_file"
        )
        exit(0)
    torch.set_grad_enabled(False)

    model_name = sys.argv[1]
    model = BertModel.from_pretrained(model_name)
    arrays = {k: v.detach() for k, v in model.named_parameters()}

    q_weight_key = 'self.query.weight'
    k_weight_key = 'self.key.weight'
    v_weight_key = 'self.value.weight'

    q_bias_key = 'self.query.bias'
    k_bias_key = 'self.key.bias'
    v_bias_key = 'self.value.bias'

    numpy_dict = {}
    for k in arrays.keys():
        if k.endswith(q_weight_key):
            v = torch.clone(
                torch.t(
                    torch.cat([
                        arrays[k],
                        arrays[k[:-len(q_weight_key)] + k_weight_key],
                        arrays[k[:-len(q_weight_key)] + v_weight_key]
                    ], 0).contiguous()).contiguous())
            numpy_dict[k[:-len(q_weight_key)] + "qkv.weight"] = v.numpy()
        elif k.endswith(q_bias_key):
            v = torch.cat([
                arrays[k], arrays[k[:-len(q_bias_key)] + k_bias_key],
                arrays[k[:-len(q_bias_key)] + v_bias_key]
            ], 0).numpy()
            numpy_dict[k[:-len(q_bias_key)] + 'qkv.bias'] = v
        elif any((k.endswith(suffix) for suffix in (k_weight_key, v_weight_key,
                                                    k_bias_key, v_bias_key))):
            continue
        elif (k.endswith("attention.output.dense.weight")
              or k.endswith("pooler.dense.weight")
              or (k.endswith("output.dense.weight")
                  or k.endswith("intermediate.dense.weight"))):
            numpy_dict[k] = torch.clone(torch.t(
                arrays[k]).contiguous()).numpy()
        else:
            numpy_dict[k] = arrays[k].numpy()
    del arrays
    del model
    numpy.savez_compressed(sys.argv[2], **numpy_dict)


if __name__ == '__main__':
    main()
