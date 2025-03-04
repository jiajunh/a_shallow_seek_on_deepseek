set -ex

python eval.py --data_names "math500" \
    --split "test" \
    --seed 0 \
    --model_name_or_path "Qwen/Qwen2.5-Math-1.5B" \
    --temperature 0 \
    --batch_size 16 \
    --prompt_type "cot" \
    --max_tokens 2048 \
    --use_safetensors \

