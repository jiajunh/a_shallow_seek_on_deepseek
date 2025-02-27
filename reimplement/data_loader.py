import os
import json
import random
import datasets

from pathlib import Path
from datasets import load_dataset, Dataset
from typing import Union, Iterable, Any


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def load_data(data_name: str, split: str, data_dir: str) -> list[Any]:
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if os.path.exists(data_file):
        data = list(load_jsonl(data_file))
        print("Load data from {}, total {} data loaded.".format(data_file, len(data)))

    if "idx" not in data[0]:
        data = [{"idx": i, **example} for i, example in enumerate(data)]
    data = sorted(data, key=lambda x: x["idx"])
    return data
        

    
