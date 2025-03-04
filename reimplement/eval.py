import os
import argparse
import torch
import time

from tqdm import tqdm

from data_loader import load_data
from utils import set_seed
from model_utils import load_hf_lm_and_tokenizer, generate_completions
from preprocess import *
from parser import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math500, aime24, amc23, minerva_math, olympiadbench", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B", type=str)
    parser.add_argument("--tokenizer_name_or_path", default="", type=str)
    parser.add_argument("--use_safetensors", action="store_true")
    # parser.add_argument("--num_test_sample", default=1, type=int)
    parser.add_argument("--temperature", default=0, type=int)
    parser.add_argument("--batch_size", default=16, type=int)

    parser.add_argument("--prompt_type", default="cot", type=str)
    # parser.add_argument("--num_shots", default=0, type=int)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--max_tokens", default=2048, type=int)

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

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

    data_list = [x.strip() for x in args.data_names.split(",")]
    results = []
    for data_name in data_list:
        results.append(main(model, tokenizer, data_name, args))

def main(model, tokenizer, data_name, args):
    print(data_name)
    data, processed_samples, out_file = prepare_data(data_name, args)
    # data has basic keys: ['idx', 'problem', 'solution', 'answer', 'subject', 'level', 'unique_id']
    print("data:", data_name, " ,remain samples:", len(data))
    if len(data):
        print("#"*100)
        print(data[0])
        print("#"*100)

    samples = []

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")
    
    for example in tqdm(data[0:1], total=len(data)):

        idx = example["idx"]
        # print(example["problem"])
        example["question"] = parse_question(example, data_name)
        # print(example["question"])
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        full_prompt = construct_prompt(example, data_name, args)
        
        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
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

    # Evaluation
    start_time = time.time()
    print("Evaluation start!")
    print("-" * 20, "Epoch", 0)

    for i in range(0, len(samples), args.batch_size):
        prompts = [s["prompt"] for s in samples[i:i+args.batch_size]]
        answers = [s["gt"] for s in samples[i:i+args.batch_size]]
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            batch_prompts=prompts,
            max_new_tokens=args.max_tokens,
            stop_id_sequences=stop_words,
        )


    print("Evaluation end!")
    end_time = time.time()



if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
    
