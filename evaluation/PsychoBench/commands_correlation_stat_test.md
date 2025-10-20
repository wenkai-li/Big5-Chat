
# IPIP-NEO 8B SFT

mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path /data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/ \
        --model_lora_path /data/user_data/wenkail/llm_personality/align/8b_lora_sft_1e-5/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

# IPIP-NEO 8B DPO

mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path /data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/ \
        --model_lora_path /data/user_data/wenkail/llm_personality/align/8b_lora_dpo_1e-5/checkpoint-2025/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_dpo_${mode}
done

# IPIP-NEO 8B Prompt V1

mode=("prompt_v1_0xxxx" "prompt_v1_1xxxx" "prompt_v1_x0xxx" "prompt_v1_x1xxx" "prompt_v1_xx0xx" "prompt_v1_xx1xx" "prompt_v1_xxx0x" "prompt_v1_xxx1x" "prompt_v1_xxxx0" "prompt_v1_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path /data/models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done