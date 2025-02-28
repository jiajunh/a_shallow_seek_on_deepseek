import torch
import tqdm
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_hf_lm_and_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: str = None,
    device_map: str = "auto", 
    load_in_half: bool = True,
    use_fast_tokenizer:bool = False,
    use_safetensors: bool = False,
) -> None:
    
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, 
        use_fast=use_fast_tokenizer, 
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.float16,
                                                device_map=device_map,
                                                trust_remote_code=True,
                                                use_safetensors=use_safetensors)

    if load_in_half:
        model = model.half()
    return model, tokenizer