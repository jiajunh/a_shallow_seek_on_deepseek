import os
import argparse

import torch

from data_loader import load_data
from utils import set_seed
from model_utils import load_hf_lm_and_tokenizer
from preprocess import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B", type=str)
    parser.add_argument("--tokenizer_name_or_path", default=None, type=str)
    parser.add_argument("--use_safetensors", default=True, type=bool)
    parser.add_argument("--num_test_sample", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=int)

    parser.add_argument("--shuffle", action="store_true")

    parser.add_argument("--model_name", )
    args = parser.parse_args()
    return args

def setup(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path = args.model_name_or_path,
        load_in_half = True,
        use_fast_tokenizer = True,
        use_safetensors = args.use_safetensors,
    )
    model.to(device)
    # print(model, tokenizer)

    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(model, tokenizer, data_name, args))

def main(model, tokenizer, data_name, args):
    prepare_data(data_name, args)
    pass

if __name__ == "__main__":
    args = parse_args()
    print(type(args))
    set_seed(args.seed)
    setup(args)
    # print(args)
    # data = load_data(args.data_names, args.split, args.data_dir)
    
