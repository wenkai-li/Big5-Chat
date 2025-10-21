from tqdm import tqdm
import pandas as pd
import json
import os
import argparse
from pathlib import Path

import openai
import numpy as np
import time
from types import SimpleNamespace

from dialogue_utils import cleanup_dialogue

class ChatGPTBaseAgent():
    def __init__(self, kwargs: dict):
        from dotenv import load_dotenv
        load_dotenv()
        openai.api_key = os.environ.get('openai_api_key')
        openai.organization = os.environ.get("openai_api_org_1")
        self.args = SimpleNamespace(**kwargs)
        self.max_retries = 5

    def generate(self, messages):
        curr_retries = 0
        while curr_retries < self.max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.args.model,
                    messages=messages
                )
                break
            except Exception as e:
                print("Error: {}".format(e))
                time.sleep(2)
                curr_retries += 1
                continue

        return completion['choices'][0].message.content.strip()


class CO3():
    def __init__(self, args):
        self.args = args

        self.set_llm_and_instruction(args)
        self.print_args()
        self.data = pd.read_csv(args.in_file).to_dict(orient='records')
    

    def set_llm_and_instruction(self, args):
        if "gpt" in args.model:
            self.llm = ChatGPTBaseAgent(args.__dict__)
        elif "dexpert" in args.model:
            import sys
            sys.path.append("../dexpert/")
            from dexpert import DExpertGenerator
            class Args:
                model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

            class ArgsExpert:
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            self.llm = DExpertGenerator(args=Args, args_expert=ArgsExpert)
        self.narrative_prompt = "Rewrite this story with more specific details in two or three sentences:"
        # self.dialogue_prompt = "Generate an in-depth conversation happening in the scene between person 1 and person 2 with multiple turns."
        self.dialogue_prompt = "Generate an in-depth conversation happening in the scene between person 1 and person 2 with less than five turns."

        self.prompt = [self.narrative_prompt, self.dialogue_prompt]
        self.prompt_suffix = "\nPerson 1:"
        self.prompt_suffix2 = "\nPerson 2:"

    def set_prompt_for_dialogue(self, text, **speakers):
        """
        Set prompt for dialogue generation with the interlocutors.
        """
        speaker_prefix = "\n" + speakers['x'] + ":"
        command_prompt = self.dialogue_prompt.replace("person 1", speakers['x'])

        # if there's PersonX and PersonY in the narrative, use them as the speakers.
        command_prompt = command_prompt.replace("person 2", speakers['y'])
        prompt = text + " " + command_prompt + speaker_prefix
        print(prompt)
        return prompt

    def print_args(self):
        # sorted_args = sorted(self.args.__dict__.items())
        print("\n=======================================")
        for idx, (k, v) in enumerate(self.args.__dict__.items()):
            if idx != 0:
                print("---------------------------------------")
            print(k, " : ", v)
        print("=======================================\n")


    def run(self):
        out_f = open(self.args.out_file, 'a')
        for current_idx, data_input in tqdm(enumerate(self.data[:10])):

            if self.args.generation_limit is not None:
                if current_idx > self.args.generation_limit:
                    break

            output = self._collect_dialogue(data_input['narrative'], data_input['PersonX'], data_input['PersonY'])

            
            info = {
                "current_idx": current_idx,
                "input": {
                    "index": data_input["Unnamed: 0"],
                    "narrative": data_input['narrative'],
                    "PersonX": data_input['PersonX'],
                    "PersonY": data_input['PersonY'],
                },
                "output": output,
            }
            json.dump(info, out_f)
            out_f.write("\n")
            out_f.flush()

    def _generate_dialogue(self, text, personx, persony):
        """
        Generate dialogue with the given narrative text.
        """
        speakers = {'x': personx, 'y': persony}
        _prompt = prompt = self.set_prompt_for_dialogue(text, **speakers)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        raw_dialogue = self.llm.generate(messages)
        result = self._parse_dialogue_output(raw_dialogue)

        result['dialogue_prompt'] = _prompt

        return result



    def _collect_dialogue(self, text, personx, persony):
        attempt_count = self.args.generation_attempt_count
        repetition_tolerance = self.args.repetition_tolerance

        dialogue = None
        generated_dialogues = []
        while dialogue is None:
            result = self._generate_dialogue(text, personx, persony)
            dialogue = result['dialogue']

            unique_utterances = set(result['utterances'])
            n_repetitive_utterances = len(result['utterances']) - len(unique_utterances)
            result['repetition'] = False # default flag
            result['suspended'] = False # default flag

            generated_dialogues.append(dialogue)

            if len(result['utterances']) < self.args.min_dialogue_turn:
                dialogue = None
                attempt_count -= 1
                print("The dialogue is too short! Attempt count left: " + str(attempt_count))
            elif len(result['speakers']) < 2:
                dialogue = None
                attempt_count -= 1
                print("There are less than two speakers! Attempt count left: " + str(attempt_count))
            elif n_repetitive_utterances > 0:
                repetition_tolerance -= 1
                print("Has " + str(n_repetitive_utterances) + " repetitive utterances! Generating the dialogue again...")
                print("Repetition tolerance:", repetition_tolerance)
                print(result['dialogue_prompt'])
                print(dialogue)
                if repetition_tolerance == 0:
                    result['repetition'] = True
                else:
                    dialogue = None
                    del generated_dialogues[-1]

            if attempt_count == 0:
                print("Tried enough!")
                result['suspended'] = True
                break

        if dialogue is None:
            # choose from the existing ones
            sorted_dialogues = sorted(generated_dialogues, key=len)
            dialogue = sorted_dialogues[-1]
            print("Choosing the longest one among the generated ones!")

        result['all_generated_dialogues'] = generated_dialogues

        return result

    def _parse_dialogue_output(self, raw_dialogue):

        # clean up dialogue
        clean_dialogue = cleanup_dialogue(raw_dialogue)
        dialogue = clean_dialogue['dialogue']
        num_responses = len(clean_dialogue['speakers'])

        result = {
            'dialogue': dialogue,
            'speakers': clean_dialogue['speakers'],
            'utterances': clean_dialogue['utterances'],
            'num_responses': num_responses,
        }

        return result


def main(args):
    soda_maker = CO3(args)
    soda_maker.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating dialogues')
    parser.add_argument("--in_file", type=str, default="soda_data/sample_10000.csv", help="The file of the sampled soda training data")
    parser.add_argument("--out_file", type=str, default="profiles/env_profiles.jsonl")
    parser.add_argument('--generation-limit',
                        type=int,
                        default=None,
                        help='the number of dialogues that this run will generate. If None, it will generate with the entire given dataset.')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-3.5-turbo-1106',
                        help='which LLM to use')
    parser.add_argument('--min-dialogue-turn',
                        type=int,
                        default=6,
                        help='minimum number of turns for a dialogue (if gpt-3 still fails to generate longer than after generation-attempt-count, it will let the dialogue be)')
    parser.add_argument('--conversation-continuation-count',
                        type=int,
                        default=1,
                        help='maximum number of attempts to continue the current conversation')
    parser.add_argument('--generation-attempt-count',
                        type=int,
                        default=2,
                        help='maximum number of attempts to generate a dialogue again')
    parser.add_argument('--repetition-tolerance',
                        type=int,
                        default=1,
                        help='maximum number of generation attempts when repetitive utterance is present in the dialogue')
    args = parser.parse_args()
    main(args)
