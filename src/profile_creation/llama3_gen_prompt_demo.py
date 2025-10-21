import json
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import json
import os
import re
from peft import PeftModel
import torch
np.random.seed(42)
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import generate_prompt
sys.path.append("../dexpert/")
import pdb
from modeling_llama import LlamaForCausalLM

def remove_double_quotes(s):
    # Remove double quotes from the beginning and end
    return re.sub(r'^"+|"+$', '', s)

prompt_demo_res = json.load(open("./prompt_demo_res.json", 'r'))

class Generator():
    def __init__(self, args):
        
        self.args = args
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model_id,
            padding_side="left",
            cache_dir=args.cache_dir
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        self.model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        
        self.model.eval()
        
    def generate_conv(self, messages):
        """
        Call the normal generate function without
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            do_sample=False
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        return result
    
    def generate(self, messages, messages_expert=None, messages_antiexpert=None, alpha=None):
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt1},
            
            {"role": "assistant", "content": response2},
            {"role": "user", "content": prompt1},
            
            {"role": "assistant", "content": response2},
        ]
        
        DExpert only has effect on assistant responses.
        """

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            alpha=alpha,
            max_new_tokens=512,
            eos_token_id=self.terminators,
            do_sample=False
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        return result
        
class CO3Sotopia():
    def __init__(self, args):
        self.args = args
        
        class Args:
            model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
            cache_dir = None
            lora = True
            
        self.model = Generator(args=Args)
        with open(args.in_file) as f:
            self.data = json.load(f)
            
        # self.data = pd.read_csv(args.in_file).to_dict(orient='records')
        
        curr_idx, total_idx = args.chunk.split("/")
        curr_idx = int(curr_idx)
        total_idx = int(total_idx)
        parts = len(self.data) // total_idx
        self.data = self.data[(curr_idx-1)*parts:curr_idx*parts]
        
    
    def process_response(self, response):
        try:
            # split by \n\n and get the middle one?
            if ":\n\n" in response:
                response = response.split(":\n\n")
                response = response[-1]
            # remove quotation marks at the beginning and the end
            response = remove_double_quotes(response)
            return response
        except Exception as e:
            print("Error during processing response", response)
            return response
        
    def generate_messages(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        return messages
    
    def generate_dialogue_turn0(self, idx, env_info, out_f):
        self.response_turn_0 = env_info
        # json.dump(env_info, out_f)
        # out_f.write("\n")
        # out_f.flush()
        
    def get_prompt(self, big_five_level):
        big_five_traits = ['o', 'c', 'e', 'a', 'n']
        level_str = ['high', 'low']
        
        instruction = ""
        for idx, level in enumerate(big_five_level):
            if level in ['0', '1']:
                level = int(level)
                instruction = prompt_demo_res[big_five_traits[idx]][level_str[level]]
                break
        print(instruction)
        return instruction
    
    def generate_dialogue_turn1(self, idx, env_info, p2_big_five, out_f):
        
        
        # generate prompt for turn 1
        # prompt_turn_1 = generate_prompt(
        #     env_info,
        #     current_turn_index=1,
        #     # p1_personality_and_values=p1_big_five,
        #     p2_personality_and_values=p2_big_five,
        #     p1_argument = self.response_turn_0,
        # )
        
        prompt_turn_1 = self.get_prompt(p2_big_five) + env_info
        
        response_turn_1 = self.process_response(self.model.generate(
            messages = self.generate_messages(prompt_turn_1),
            alpha = self.args.alpha,
        ))
        
        result_info = {
            "env_idx": idx,
            "env_info": env_info,
            "personality": " ".join(map(str, p2_big_five)),
            "turn": 1,
            "prompt": prompt_turn_1,
            "response": response_turn_1,
        }
        json.dump(result_info, out_f)
        out_f.write("\n")
        out_f.flush()
    
    def run(self):
        if os.path.exists(self.args.out_file):
            in_f = open(self.args.out_file, 'r')
            in_data = [json.loads(line) for line in in_f.readlines()]
            completed_idx = len(in_data)
        else:
            completed_idx = 0
        
        out_f = open(self.args.out_file, 'a')
        p2_big_five_ref = [-1, -1, -1, -1, -1]
        p2_big_five_abbr = {'o': 0, 'c': 1, 'e': 2, 'a': 3, 'n': 4}
        
        p2_big_five_high, p2_big_five_low = p2_big_five_ref.copy(), p2_big_five_ref.copy()
        p2_big_five_high[p2_big_five_abbr[self.args.person_trait]] = 0 # high
        p2_big_five_low[p2_big_five_abbr[self.args.person_trait]] = 1 # low
        
        for idx, env_info in tqdm(enumerate(self.data[completed_idx:])):
            print("Processing: ", idx)
            self.generate_dialogue_turn0(idx, env_info['input'], out_f)
            self.generate_dialogue_turn1(idx, env_info['input'], p2_big_five_high, out_f)
            self.generate_dialogue_turn1(idx, env_info['input'], p2_big_five_low, out_f)
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating dialogues')
    parser.add_argument("--in_file", type=str, default="./soda_data/test.json", help="The file of the sampled soda training data")
    parser.add_argument("--out_file", type=str, default="./profiles/env_profiles.jsonl")
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--person_trait", type=str, choices=['o', 'c', 'e', 'a', 'n'])
    parser.add_argument("--chunk", type=str, default="1/50")
    args = parser.parse_args()
    soda_maker = CO3Sotopia(args)
    soda_maker.run()