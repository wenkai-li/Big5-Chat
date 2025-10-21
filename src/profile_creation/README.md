We need to generate 5000 environment profiles (Actually 10k to avoid risks)


- `env_profiles_w_generator.jsonl`: llama3 70B without bfi prompts + trained llama3 8B generator

- `env_profiles.jsonl`: llama3 70B with bfi prompts + trained llama3 8B generator

<!-- # done -->
<!-- python3 llama3_gen.py --out_file ./profiles/env_profiles_1_alpha0.jsonl --alpha 0.0 --chunk 1/5
python3 llama3_gen.py --out_file ./env_profiles_1_alpha1.jsonl --alpha 1.0 --chunk 1/5 -->

# running
python3 llama3_gen.py --out_file ./env_profiles_o_1.jsonl --alpha 0.5 --chunk 1/2 --person_trait o
python3 llama3_gen.py --out_file ./env_profiles_o_2.jsonl --alpha 0.5 --chunk 2/2 --person_trait o
python3 llama3_gen.py --out_file ./env_profiles_c_1.jsonl --alpha 0.5 --chunk 1/2 --person_trait c
python3 llama3_gen.py --out_file ./env_profiles_c_2.jsonl --alpha 0.5 --chunk 2/2 --person_trait c
python3 llama3_gen.py --out_file ./env_profiles_e_1.jsonl --alpha 0.5 --chunk 1/2 --person_trait e
python3 llama3_gen.py --out_file ./env_profiles_e_2.jsonl --alpha 0.5 --chunk 2/2 --person_trait e
python3 llama3_gen.py --out_file ./env_profiles_a_1.jsonl --alpha 0.5 --chunk 1/2 --person_trait a
python3 llama3_gen.py --out_file ./env_profiles_a_2.jsonl --alpha 0.5 --chunk 2/2 --person_trait a
python3 llama3_gen.py --out_file ./env_profiles_n_1.jsonl --alpha 0.5 --chunk 1/2 --person_trait n
python3 llama3_gen.py --out_file ./env_profiles_n_2.jsonl --alpha 0.5 --chunk 2/2 --person_trait n


<!-- 
CUDA_VISIBLE_DEVICES=0 python llama3_gen_finetune.py --out_file ./finetune_soda/sft_n_1.jsonl --person_trait n --model_path ./align/70b_gptq_lora_sft_1e-5/

CUDA_VISIBLE_DEVICES=1 python llama3_gen_finetune.py --out_file ./finetune_soda/dpo_n_1.jsonl --person_trait n --model_path ./align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/

CUDA_VISIBLE_DEVICES=0 python llama3_gen_finetune.py --out_file ./finetune_soda/dpo_o_1.jsonl --person_trait o --model_path ./align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/

CUDA_VISIBLE_DEVICES=1 python llama3_gen_finetune.py --out_file ./finetune_soda/dpo_c_1.jsonl --person_trait c --model_path ./align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/

CUDA_VISIBLE_DEVICES=0 python llama3_gen_finetune.py --out_file ./finetune_soda/dpo_e_1.jsonl --person_trait e --model_path ./align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/

CUDA_VISIBLE_DEVICES=1 python llama3_gen_finetune.py --out_file ./finetune_soda/dpo_a_1.jsonl --person_trait a --model_path ./align/70b_gptq_lora_dpo_1e-5/llama3_70b_gptq/checkpoint-2025/

## general prommpt chat o c e a

CUDA_VISIBLE_DEVIVES=0,1 python llama3_gen_prompt_chat.py --out_file ./finetune_soda/prompt_chat_o_1.jsonl --person_trait o


CUDA_VISIBLE_DEVIVES=0,1 python llama3_gen_prompt_chat.py --out_file ./finetune_soda/prompt_chat_c_1.jsonl --person_trait c

CUDA_VISIBLE_DEVIVES=0,1 python llama3_gen_prompt_chat.py --out_file ./finetune_soda/prompt_chat_e_1.jsonl --person_trait e

CUDA_VISIBLE_DEVIVES=0,1 python llama3_gen_prompt_chat.py --out_file ./finetune_soda/prompt_chat_a_1.jsonl --person_trait a

CUDA_VISIBLE_DEVIVES=0,1 python llama3_gen_prompt_chat.py --out_file ./finetune_soda/prompt_chat_n_1.jsonl --person_trait n -->