# BFI 70B prompt_keywords

mode=("prompt_keywords_0xxxx" "prompt_keywords_1xxxx" "prompt_keywords_x0xxx" "prompt_keywords_x1xxx" "prompt_keywords_xx0xx" "prompt_keywords_xx1xx" "prompt_keywords_xxx0x" "prompt_keywords_xxx1x" "prompt_keywords_xxxx0" "prompt_keywords_xxxx1")
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

# BFI 8B prompt_keywords

mode=("prompt_keywords_0xxxx" "prompt_keywords_1xxxx" "prompt_keywords_x0xxx" "prompt_keywords_x1xxx" "prompt_keywords_xx0xx" "prompt_keywords_xx1xx" "prompt_keywords_xxx0x" "prompt_keywords_xxx1x" "prompt_keywords_xxxx0" "prompt_keywords_xxxx1")
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

# IPIP 70B prompt_keywords

mode=("prompt_keywords_0xxxx" "prompt_keywords_1xxxx" "prompt_keywords_x0xxx" "prompt_keywords_x1xxx" "prompt_keywords_xx0xx" "prompt_keywords_xx1xx" "prompt_keywords_xxx0x" "prompt_keywords_xxx1x" "prompt_keywords_xxxx0" "prompt_keywords_xxxx1")
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

# IPIP 8B prompt_keywords

mode=("prompt_keywords_0xxxx" "prompt_keywords_1xxxx" "prompt_keywords_x0xxx" "prompt_keywords_x1xxx" "prompt_keywords_xx0xx" "prompt_keywords_xx1xx" "prompt_keywords_xxx0x" "prompt_keywords_xxx1x" "prompt_keywords_xxxx0" "prompt_keywords_xxxx1")
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
