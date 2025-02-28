import os 

from datetime import datetime
from data_loader import *

from utils import load_jsonl
from typing import Tuple, Any

def prepare_data(data_name: str, args) -> Tuple[list[Any], list[Any], str]: 
    data = load_data(data_name, args.split, args.data_dir)
    if args.shuffle:
        random.shuffle(data)

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    data = [example for example in data if example["idx"] not in processed_idxs]
    return data, processed_samples, out_file