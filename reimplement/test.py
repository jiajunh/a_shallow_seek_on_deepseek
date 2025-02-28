import os
import argparse
import torch

from tqdm import tqdm

from data_loader import load_data
from utils import set_seed
from model_utils import load_hf_lm_and_tokenizer
from preprocess import *
from parser import *

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
    parser.add_argument("--overwrite", action="store_true")

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
    data, processed_samples, out_file = prepare_data(data_name, args)
    # data has basic keys: ['idx', 'problem', 'solution', 'answer', 'subject', 'level', 'unique_id']
    print("data:", data_name, " ,remain samples:", len(data))
    if len(data):
        print("#"*100)
        print(data[0])
        print("#"*100)
    
    samples = []
    for example in tqdm(data[0:1], total=len(data)):
        # print(example.keys())

        idx = example["idx"]
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        sample = {
            "idx": idx,
            "question": example["question"],
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
    # print(args)
    # data = load_data(args.data_names, args.split, args.data_dir)
    
