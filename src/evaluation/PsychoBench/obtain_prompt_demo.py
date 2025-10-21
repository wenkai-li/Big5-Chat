import time
import json
import pandas as pd

seed = 42

if __name__ == "__main__":
    big_five_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    levels = ['high', 'low']
    n_examples = 10
    
    out_f = open("prompt_demo_res.json", 'w')
    res_data = {}
    for idx, trait in enumerate(['o', 'c', 'e', 'a', 'n']):
        res_data[trait] = {}
        for level in levels:
            personality_str = ["-1", "-1", "-1", "-1", "-1"]
            personality_str[idx] = '0' if level == 'high' else "1"
            personality_str = " ".join(personality_str)
            
            # Here please replace with the hugging data sample logics
            df = pd.DataFrame().from_records([json.loads(i) for i in open(f"./profiles/env_profiles_{trait}_1.jsonl", 'r').readlines()] + [json.loads(i) for i in open(f"./profiles/env_profiles_{trait}_2.jsonl", 'r').readlines()])
            
            df = df.loc[df['personality'] == personality_str]
            examples = df.sample(n = n_examples, random_state=seed)['response'].tolist()
            examples = "\n```\n" + "\n".join([f'Example {i}: "{example}"' for i, example in enumerate(examples)]) + "\n```\n\n"
            
            instruction = f"""Here are {n_examples} examples of how people like you have responded in different situations. Pay attention to how they approach communication and problem-solving.

""" + examples
            
            res_data[trait][level] = instruction
    
    json.dump(res_data, out_f)
    out_f.flush()