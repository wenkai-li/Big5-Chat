# BFI 70B prompt_v5

mode=("prompt_v5_0xxxx" "prompt_v5_1xxxx" "prompt_v5_x0xxx" "prompt_v5_x1xxx" "prompt_v5_xx0xx" "prompt_v5_xx1xx" "prompt_v5_xxx0x" "prompt_v5_xxx1x" "prompt_v5_xxxx0" "prompt_v5_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

# BFI 8B prompt_v5

mode=("prompt_v5_0xxxx" "prompt_v5_1xxxx" "prompt_v5_x0xxx" "prompt_v5_x1xxx" "prompt_v5_xx0xx" "prompt_v5_xx1xx" "prompt_v5_xxx0x" "prompt_v5_xxx1x" "prompt_v5_xxxx0" "prompt_v5_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 8b_${mode}
done

# IPIP 70B prompt_v5

mode=("prompt_v5_0xxxx" "prompt_v5_1xxxx" "prompt_v5_x0xxx" "prompt_v5_x1xxx" "prompt_v5_xx0xx" "prompt_v5_xx1xx" "prompt_v5_xxx0x" "prompt_v5_xxx1x" "prompt_v5_xxxx0" "prompt_v5_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

# IPIP 8B prompt_v5

mode=("prompt_v5_0xxxx" "prompt_v5_1xxxx" "prompt_v5_x0xxx" "prompt_v5_x1xxx" "prompt_v5_xx0xx" "prompt_v5_xx1xx" "prompt_v5_xxx0x" "prompt_v5_xxx1x" "prompt_v5_xxxx0" "prompt_v5_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 8b_${mode}
done
