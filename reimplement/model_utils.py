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


@torch.no_grad()
def generate_completions(model, 
                         tokenizer,
                         batch_prompts, 
                         max_new_tokens, 
                         stop_id_sequences,
                         add_special_tokens=True,
                         **generation_kwargs):

    device = model.device
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    
    tokenized_prompts = tokenizer(batch_prompts, 
                                  padding="longest", 
                                  return_tensors="pt", 
                                  add_special_tokens=add_special_tokens)
    
    batch_input_ids = tokenized_prompts.input_ids.to(device)
    attention_mask = tokenized_prompts.attention_mask.to(device)

    # print(generation_kwargs)

    batch_outputs = model.generate(
        input_ids=batch_input_ids,
        attention_mask=attention_mask,
        # stopping_criteria=StoppingCriteriaList([stop_criteria]),
        **generation_kwargs
    )
    print(batch_outputs)

    