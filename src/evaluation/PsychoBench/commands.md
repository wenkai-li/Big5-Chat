# BFI
## 70B

### Direct
python3 run_psychobench.py \
    --model llama3_70b \
    --model_mode direct \
    --model_path meta-llama/Meta-Llama-3-70B-Instruct \
    --questionnaire BFI \
    --shuffle-count 0 \
    --test-count 5 \
    --mode analysis \
    --name-exp direct_debug

### SFT

mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --model_lora_path align/70b_gptq_lora_sft_1e-5/ \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_${mode}
done

### DPO

mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --model_lora_path align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/ \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_dpo_${mode}
done

### Prompt
mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_${mode}
done

### Prompt_demo
mode=("prompt_demo_0xxxx" "prompt_demo_1xxxx" "prompt_demo_x0xxx" "prompt_demo_x1xxx" "prompt_demo_xx0xx" "prompt_demo_xx1xx" "prompt_demo_xxx0x" "prompt_demo_xxx1x" "prompt_demo_xxxx0" "prompt_demo_xxxx1")
for mode in "${mode[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

### Prompt_demo_sampling
mode=("prompt_demo_sampling_0xxxx" "prompt_demo_sampling_1xxxx" "prompt_demo_sampling_x0xxx" "prompt_demo_sampling_x1xxx" "prompt_demo_sampling_xx0xx" "prompt_demo_sampling_xx1xx" "prompt_demo_sampling_xxx0x" "prompt_demo_sampling_xxx1x" "prompt_demo_sampling_xxxx0" "prompt_demo_sampling_xxxx1")
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

mode=("prompt_demo_sampling_0xxxx")
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

### Prompt_inst
mode=("prompt_inst_0xxxx" "prompt_inst_1xxxx" "prompt_inst_x0xxx" "prompt_inst_x1xxx" "prompt_inst_xx0xx" "prompt_inst_xx1xx" "prompt_inst_xxx0x" "prompt_inst_xxx1x" "prompt_inst_xxxx0" "prompt_inst_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_${mode}
done

### Prompt_llm_description
mode=("prompt_llm_description_0xxxx" "prompt_llm_description_1xxxx" "prompt_llm_description_x0xxx" "prompt_llm_description_x1xxx" "prompt_llm_description_xx0xx" "prompt_llm_description_xx1xx" "prompt_llm_description_xxx0x" "prompt_llm_description_xxx1x" "prompt_llm_description_xxxx0" "prompt_llm_description_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_${mode}
done

### Prompt_
mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct/ \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

## 8B

### Direct
python3 run_psychobench.py \
    --model llama3_8b \
    --model_mode direct \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --questionnaire BFI \
    --shuffle-count 0 \
    --test-count 5 \
    --mode analysis \
    --name-exp direct_debug_8b

### SFT
mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --model_lora_path align/8b_lora_sft_1e-5/ \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

### DPO
mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --model_lora_path align/8b_lora_dpo_1e-5/checkpoint-2025/ \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_dpo_${mode}
done

### Prompt
mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

### Prompt chat
mode=("prompt_demo_0xxxx" "prompt_demo_1xxxx" "prompt_demo_x0xxx" "prompt_demo_x1xxx" "prompt_demo_xx0xx" "prompt_demo_xx1xx" "prompt_demo_xxx0x" "prompt_demo_xxx1x" "prompt_demo_xxxx0" "prompt_demo_xxxx1")
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

### Prompt_demo_sampling
mode=("prompt_demo_sampling_0xxxx" "prompt_demo_sampling_1xxxx" "prompt_demo_sampling_x0xxx" "prompt_demo_sampling_x1xxx" "prompt_demo_sampling_xx0xx" "prompt_demo_sampling_xx1xx" "prompt_demo_sampling_xxx0x" "prompt_demo_sampling_xxx1x" "prompt_demo_sampling_xxxx0" "prompt_demo_sampling_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 8b_${mode}
done


### Prompt v1
mode=("prompt_inst_0xxxx" "prompt_inst_1xxxx" "prompt_inst_x0xxx" "prompt_inst_x1xxx" "prompt_inst_xx0xx" "prompt_inst_xx1xx" "prompt_inst_xxx0x" "prompt_inst_xxx1x" "prompt_inst_xxxx0" "prompt_inst_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

### Prompt v4
mode=("prompt_llm_description_0xxxx" "prompt_llm_description_1xxxx" "prompt_llm_description_x0xxx" "prompt_llm_description_x1xxx" "prompt_llm_description_xx0xx" "prompt_llm_description_xx1xx" "prompt_llm_description_xxx0x" "prompt_llm_description_xxx1x" "prompt_llm_description_xxxx0" "prompt_llm_description_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire BFI \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

### Prompt_
mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
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

# IPIP-NEO

## 70B

### Direct
python3 run_psychobench.py \
    --model llama3_70b \
    --model_mode direct \
    --model_path meta-llama/Meta-Llama-3-70B-Instruct \
    --questionnaire IPIP-NEO \
    --shuffle-count 0 \
    --test-count 5 \
    --mode analysis \
    --name-exp direct_debug

### SFT
mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --model_lora_path align/70b_gptq_lora_sft_1e-5/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_${mode}
done

### DPO
mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --model_lora_path align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_dpo_${mode}
done

### Prompt
mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
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

### Prompt v1
mode=("prompt_inst_0xxxx" "prompt_inst_1xxxx" "prompt_inst_x0xxx" "prompt_inst_x1xxx" "prompt_inst_xx0xx" "prompt_inst_xx1xx" "prompt_inst_xxx0x" "prompt_inst_xxx1x" "prompt_inst_xxxx0" "prompt_inst_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 70b_${mode}
done

# Prompt chat
mode=("prompt_demo_0xxxx" "prompt_demo_1xxxx" "prompt_demo_x0xxx" "prompt_demo_x1xxx" "prompt_demo_xx0xx")
for mode in "${mode[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

mode=("prompt_demo_xx1xx" "prompt_demo_xxx0x" "prompt_demo_xxx1x" "prompt_demo_xxxx0" "prompt_demo_xxxx1")
for mode in "${mode[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

### Prompt_demo_sampling
mode=("prompt_demo_sampling_0xxxx" "prompt_demo_sampling_1xxxx" "prompt_demo_sampling_x0xxx" "prompt_demo_sampling_x1xxx" "prompt_demo_sampling_xx0xx" "prompt_demo_sampling_xx1xx" "prompt_demo_sampling_xxx0x" "prompt_demo_sampling_xxx1x" "prompt_demo_sampling_xxxx0" "prompt_demo_sampling_xxxx1")
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

mode=("prompt_demo_sampling_xx1xx")
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

### Prompt v4
mode=("prompt_llm_description_0xxxx" "prompt_llm_description_1xxxx" "prompt_llm_description_x0xxx" "prompt_llm_description_x1xxx" "prompt_llm_description_xx0xx" "prompt_llm_description_xx1xx" "prompt_llm_description_xxx0x" "prompt_llm_description_xxx1x" "prompt_llm_description_xxxx0" "prompt_llm_description_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

### Prompt_

mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_70b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-70B-Instruct/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --name-exp 70b_${mode}
done

## 8B

### Direct
python3 run_psychobench.py \
    --model llama3_8b \
    --model_mode direct \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --questionnaire IPIP-NEO \
    --shuffle-count 0 \
    --test-count 5 \
    --mode analysis \
    --name-exp direct_debug_8b

### SFT
mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --model_lora_path align/8b_lora_sft_1e-5/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

### DPO
mode=("train_0xxxx" "train_1xxxx" "train_x0xxx" "train_x1xxx" "train_xx0xx" "train_xx1xx" "train_xxx0x" "train_xxx1x" "train_xxxx0" "train_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --model_lora_path align/8b_lora_dpo_1e-5/checkpoint-2025/ \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_dpo_${mode}
done

### Prompt v1
mode=("prompt_inst_0xxxx" "prompt_inst_1xxxx" "prompt_inst_x0xxx" "prompt_inst_x1xxx" "prompt_inst_xx0xx" "prompt_inst_xx1xx" "prompt_inst_xxx0x" "prompt_inst_xxx1x" "prompt_inst_xxxx0" "prompt_inst_xxxx1")
for mode in "${mode[@]}"; do
    python3 run_psychobench.py \
        --model llama3_8b \
        --model_mode ${mode} \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --questionnaire IPIP-NEO \
        --shuffle-count 0 \
        --test-count 5 \
        --mode analysis \
        --name-exp 8b_${mode}
done

### Prompt
mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
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

### Prompt v4
mode=("prompt_llm_description_0xxxx" "prompt_llm_description_1xxxx" "prompt_llm_description_x0xxx" "prompt_llm_description_x1xxx" "prompt_llm_description_xx0xx" "prompt_llm_description_xx1xx" "prompt_llm_description_xxx0x" "prompt_llm_description_xxx1x" "prompt_llm_description_xxxx0" "prompt_llm_description_xxxx1")
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

### Prompt_

mode=("prompt_0xxxx" "prompt_1xxxx" "prompt_x0xxx" "prompt_x1xxx" "prompt_xx0xx" "prompt_xx1xx" "prompt_xxx0x" "prompt_xxx1x" "prompt_xxxx0" "prompt_xxxx1")
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