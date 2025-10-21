import time
import openai
import json

# step 1
# data_q = json.load(open("questionnaires.json", 'r'))
# data = {}
# for q in data_q:
#     if q['name'] in ['BFI', 'IPIP-NEO']:
#         data[q['name']] = {
#             "questions": q['questions'],
#             "reverse": [str(i) for i in q['reverse']],
#             "categories": q['categories'],
#         }
        
# big_five_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

# data_v4 = {}
# for trait in big_five_traits:
#     data_v4[trait] = {"high": [], "low": []}
#     for name in ['BFI', 'IPIP-NEO']:
#         for category in data[name]['categories']:
#             if category['cat_name'] == trait:
#                 cat_lst = [str(i) for i in category['cat_questions']]
#                 for cat_item in cat_lst:
#                     if cat_item in data[name]['reverse']:
#                         data_v4[trait]['low'].append(data[name]['questions'][cat_item])
#                     else:
#                         data_v4[trait]['high'].append(data[name]['questions'][cat_item])
#                 break

# out_f = open("prompt_v4_questionnaires.json", 'w')
# json.dump(data_v4, out_f)
# out_f.flush()

# step 2: prompt the same LLM
data = json.load(open("prompt_v4_questionnaires.json", 'r'))
prompt_gen_descriptions_v4 = """You are a person with {level} {trait}. Write a 50-word paragraph describing character traits or behaviors that reflect your {trait}. Ensure the generated description is relevant to this trait but does not repeat any wording or behaviors from the following two lists {trait}:

{level_upper} {trait}:
{behavior_set}

{level_rev_upper} {trait}:
{behavior_set_rev}

Please provide the description directly, without explanations or additional context."""

class GPT:
    def __init__(self, model_id):
        with open("/home/jiaruil5/openai_key.txt", 'r') as file:
            content = []
            for i in file.readlines():
                content.append(i.strip())
            openai.api_key = content[0]
            openai.organization = content[1]
        
        
        self.client = openai.chat.completions
        self.model_id = model_id
        self.max_retries = 5
        
    
    def generate(self, messages, **inference_config):
        curr_retries = 0
        while True:
            try:
                response = self.client.create(
                    model = self.model_id,
                    messages = messages,
                    temperature=0.0,
                    max_tokens=2048,
                    **inference_config,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
                if curr_retries >= self.max_retries:
                    print("EXITING...")
                    exit(1)
                else:
                    print("RETRYING...")
                    curr_retries += 1
                    time.sleep(5)



if __name__ == "__main__":
    gpt = GPT("gpt-4o")
    big_five_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    levels = ['high', 'low']
    
    out_f = open("prompt_v4_res.json", 'w')
    res_data = {}
    for trait in big_five_traits:
        res_data[trait] = {}
        for idx, level in enumerate(levels):
            level_rev = levels[1 - idx]
            prompt = prompt_gen_descriptions_v4.format(
                level=level,
                trait=trait.lower(),
                level_upper=level.capitalize(),
                level_rev_upper=level_rev.capitalize(),
                behavior_set="\n".join([f"{str(i+1)}. {item}" for i, item in enumerate(data[trait][level])]),
                behavior_set_rev="\n".join([f"{str(i+1)}. {item}" for i, item in enumerate(data[trait][level_rev])]),
            )
            
            print("Prompt:")
            print(prompt)
            
            res = gpt.generate([
                {"role": "user", "content": prompt}
            ])
            res_data[trait][level] = res
            
            print("Response:")
            print(res)
    
    json.dump(res_data, out_f)
    out_f.flush()