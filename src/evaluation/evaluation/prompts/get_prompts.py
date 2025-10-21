import json
import pandas as pd

big_five_descriptions = [
    [   
        # You are a high openness person
        # generate another description
        # https://ipip.ori.org/MiniIPIPKey.htm 	Given: Am the life of the party.
            # Test: Talk to a lot of different people at parties.
        "You are an open person with a vivid imagination and a passion for the arts. You are emotionally expressive and have a strong sense of adventure. Your intellect is sharp and your views are liberal. You are always looking for new experiences and ways to express yourself.",
        "You are a closed person, and it shows in many ways. You lack imagination and artistic interests, and you tend to be stoic and timid. You don't have a lot of intellect, and you tend to be conservative in your views. You don't take risks and you don't like to try new things. You prefer to stay in your comfort zone and don't like to venture out. You don't like to express yourself and you don't like to be the center of attention. You don't like to take chances and you don't like to be challenged. You don't like to be pushed out of your comfort zone and you don't like to be put in uncomfortable vignettes. You prefer to stay in the background and not draw attention to yourself."
    ],
    [
        "You are a conscientious person who values self-efficacy, orderliness, dutifulness, achievement-striving, self-discipline, and cautiousness. You take pride in your work and strive to do your best. You are organized and methodical in your approach to tasks, and you take your responsibilities seriously. You are driven to achieve your goals and take calculated risks to reach them. You are disciplined and have the ability to stay focused and on track. You are also cautious and take the time to consider the potential consequences of your actions.",
        "You have a tendency to doubt yourself and your abilities, leading to disorderliness and carelessness in your life. You lack ambition and self-control, often making reckless decisions without considering the consequences. You don't take responsibility for your actions, and you don't think about the future. You're content to live in the moment, without any thought of the future."
    ],
    [
        "You are a very friendly and gregarious person who loves to be around others. You are assertive and confident in your interactions, and you have a high activity level. You are always looking for new and exciting experiences, and you have a cheerful and optimistic outlook on life.",
        "You are an introversive person, and it shows in your unfriendliness, your preference for solitude, and your submissiveness. You tend to be passive and calm, and you take life seriously. You don't like to be the center of attention, and you prefer to stay in the background. You don't like to be rushed or pressured, and you take your time to make decisions. You are content to be alone and enjoy your own company."
    ],
    [
        "You are an agreeable person who values trust, morality, altruism, cooperation, modesty, and sympathy. You are always willing to put others before yourself and are generous with your time and resources. You are humble and never boast about your accomplishments. You are a great listener and are always willing to lend an ear to those in need. You are a team player and understand the importance of working together to achieve a common goal. You are a moral compass and strive to do the right thing in all vignettes. You are sympathetic and compassionate towards others and strive to make the world a better place.",
        "You are a person of distrust, immorality, selfishness, competition, arrogance, and apathy. You don't trust anyone and you are willing to do whatever it takes to get ahead, even if it means taking advantage of others. You are always looking out for yourself and don't care about anyone else. You thrive on competition and are always trying to one-up everyone else. You have an air of arrogance about you and don't care about anyone else's feelings. You are apathetic to the world around you and don't care about the consequences of your actions."
    ],
    [
        "You feel like you're constantly on edge, like you can never relax. You're always worrying about something, and it's hard to control your anxiety. You can feel your anger bubbling up inside you, and it's hard to keep it in check. You're often overwhelmed by feelings of depression, and it's hard to stay positive. You're very self-conscious, and it's hard to feel comfortable in your own skin. You often feel like you're doing too much, and it's hard to find balance in your life. You feel vulnerable and exposed, and it's hard to trust others.",
        "You are a stable person, with a calm and contented demeanor. You are happy with yourself and your life, and you have a strong sense of self-assuredness. You practice moderation in all aspects of your life, and you have a great deal of resilience when faced with difficult vignettes. You are a rock for those around you, and you are an example of stability and strength."
    ]
]

def get_prompting_instruction(levels: str) -> str:
    """
    xxxxx: ocean, 0 represents high level, 1 represents low level
    - Should be 0xxxx, 1xxxx, x0xxx, x1xxx, etc.
    """
    instruction = "Imagine you are someone that fits this description: "
    assert len(levels) == 5
    assert levels.count("x") == 4
    assert '0' in levels or '1' in levels
    
    level_str = ""
    for idx, level in enumerate(levels):
        if level in ['0', '1']:
            level = int(level)
            level_str = big_five_descriptions[idx][level]
            break
    
    instruction += level_str
    return instruction

def get_prompting_instruction_inst(levels: str) -> str:
    """
    xxxxx: ocean, 0 represents high level, 1 represents low level
    - Should be 0xxxx, 1xxxx, x0xxx, x1xxx, etc.
    """
    assert len(levels) == 5
    assert levels.count("x") == 4
    assert '0' in levels or '1' in levels

    instruction = "You are a person with "
    big_five_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    level_str = ['high', 'low']

    for idx, level in enumerate(levels):
        if level in ['0', '1']:
            level = int(level)
            instruction += f"{level_str[level]} {big_five_traits[idx]}."
            break

    return instruction
    

import json
prompt_llm_description_res = json.load(open("../../PsychoBench/prompt_llm_description_res.json", 'r'))

def get_prompting_instruction_llm_description(levels: str) -> str:
    """
    xxxxx: ocean, 0 represents high level, 1 represents low level
    - Should be 0xxxx, 1xxxx, x0xxx, x1xxx, etc.
    """
    assert len(levels) == 5
    assert levels.count("x") == 4
    assert '0' in levels or '1' in levels
    instruction = """Imagine you are someone that fits this description:
```
{description}
```

"""
    big_five_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    level_str = ['high', 'low']
    
    new_instruction = ""
    for idx, level in enumerate(levels):
        if level in ['0', '1']:
            level = int(level)
            new_instruction = instruction.format(description=prompt_llm_description_res[big_five_traits[idx].capitalize()][level_str[level]])
            break
    print(new_instruction)
    return new_instruction

prompt_demo_res = json.load(open("../../PsychoBench/prompt_demo_res.json", 'r'))

def get_prompting_instruction_demo(levels: str) -> str:
    """
    xxxxx: ocean, 0 represents high level, 1 represents low level
    - Should be 0xxxx, 1xxxx, x0xxx, x1xxx, etc.
    """
    assert len(levels) == 5
    assert levels.count("x") == 4
    assert '0' in levels or '1' in levels
    
    big_five_traits = ['o', 'c', 'e', 'a', 'n']
    level_str = ['high', 'low']
    
    instruction = ""
    for idx, level in enumerate(levels):
        if level in ['0', '1']:
            level = int(level)
            instruction = prompt_demo_res[big_five_traits[idx]][level_str[level]]
            break
    print(instruction)
    return instruction

class PromptDemoSampler:
    def __init__(self):
        self.big_five_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        levels = ['high', 'low']
        self.n_examples = 10
        self.df = {}
        for idx, trait in enumerate(['o', 'c', 'e', 'a', 'n']):
            self.df[trait] = []
            for level in levels:
                personality_str = ["-1", "-1", "-1", "-1", "-1"]
                personality_str[idx] = '0' if level == 'high' else "1"
                personality_str = " ".join(personality_str)
                
                tmp_df = pd.DataFrame().from_records([json.loads(i) for i in open(f"/data/user_data/wenkail/llm_personality/profiles/env_profiles_{trait}_1.jsonl", 'r').readlines()] + [json.loads(i) for i in open(f"/data/user_data/wenkail/llm_personality/profiles/env_profiles_{trait}_2.jsonl", 'r').readlines()])
                                  
                self.df[trait].append(tmp_df.loc[tmp_df['personality'] == personality_str])
        
        
        
    def get_prompt_demo_sample(self, in_trait, in_level):
        examples = self.df[in_trait][in_level].sample(n = self.n_examples)['response'].tolist()

        examples = "\n```\n" + "\n".join([f'Example {i}: "{example}"' for i, example in enumerate(examples)]) + "\n```\n\n"

        instruction = f"""Here are {self.n_examples} examples of how people like you have responded in different situations. Pay attention to how they approach communication and problem-solving.

""" + examples

        return instruction


def get_prompting_instruction_demo_sampling(levels: str) -> str:
    """
    xxxxx: ocean, 0 represents high level, 1 represents low level
    - Should be 0xxxx, 1xxxx, x0xxx, x1xxx, etc.
    """
    assert len(levels) == 5
    assert levels.count("x") == 4
    assert '0' in levels or '1' in levels
    
    big_five_traits = ['o', 'c', 'e', 'a', 'n']
    level_str = ['high', 'low']

    prompt_demo_sampler = PromptDemoSampler()

    instruction = ""
    for idx, level in enumerate(levels):
        if level in ['0', '1']:
            level = int(level)
            instruction = prompt_demo_sampler.get_prompt_demo_sample(big_five_traits[idx], level)
            break
    print(instruction)
    return instruction

def get_trained_instruction(levels: str) -> str:
    """
    xxxxx: ocean, 0 represents high level, 1 represents low level
    - Should be 0xxxx, 1xxxx, x0xxx, x1xxx, etc.
    """
    instruction = "You are a helpful assistant with the following Big Five personality traits: "
    assert len(levels) == 5
    assert levels.count("x") == 4
    assert '0' in levels or '1' in levels
    
    level_str = ""
    for idx, level in enumerate(levels):
        if level in ['0', '1']:
            if idx == 0:
                level_str += "Openness"
            elif idx == 1:
                level_str += "Conscientiousness"
            elif idx == 2:
                level_str += "Extraversion"
            elif idx == 3:
                level_str += "Agreeableness"
            elif idx ==4:
                level_str += "Neuroticism"
            
            level_str += " - "
            if level == "0":
                level_str += "high"
            elif level == "1":
                level_str += "low"

            break
    
    instruction += level_str
    return instruction