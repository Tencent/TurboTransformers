from transformers.modeling_bert import BertModel
import sys
import numpy
import torch


def main():
    if len(sys.argv) != 3:
        print("Usage: \n"
              "    convert_huggingface_bert_to_npz model_name output_file")
        exit(0)
    torch.set_grad_enabled(False)

    model_name = sys.argv[1]
    model = BertModel.from_pretrained(model_name)
    arrays = {k: v.detach().numpy() for k, v in model.named_parameters()}
    del model
    numpy.savez_compressed(sys.argv[2], **arrays)


if __name__ == '__main__':
    main()
