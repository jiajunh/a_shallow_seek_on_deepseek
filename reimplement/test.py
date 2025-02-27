import os
import argparse

from data_loader import load_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    data = load_data(args.data_names, args.split, args.data_dir)
    
