import os
import random
import datasets

from datasets import load_dataset, Dataset
from utils import load_jsonl
from typing import Union, Iterable, Any



def load_data(data_name: str, split: str, data_dir: str) -> list[Any]:
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    print(data_file)
    if os.path.exists(data_file):
        data = list(load_jsonl(data_file))
        print("Load data from {}, total {} data loaded.".format(data_file, len(data)))

    if "idx" not in data[0]:
        data = [{"idx": i, **example} for i, example in enumerate(data)]
    data = sorted(data, key=lambda x: x["idx"])
    return data
        

    
