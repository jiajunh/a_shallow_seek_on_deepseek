from datetime import datetime
from data_loader import *

def prepare_data(data_name: str, args):
    data = load_data(data_name, args.split, args.data_dir)
    if args.shuffle:
        random.shuffle(data)

    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)
