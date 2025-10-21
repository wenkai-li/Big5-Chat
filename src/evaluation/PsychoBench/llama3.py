import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel



class LLAMA3():
    def __init__(self, args):
        """
        args: model, model_mode, model_path, model_lora_path, cache_dir
        - parser.add_argument('--model', required=True, type=str, default='text-davinci-003', help='The name of the model to test')
        - parser.add_argument('--cache_dir', type=str, default=None)
        - parser.add_argument('--model_mode', type=str, default='direct', help='direct, train_xxxxx, or prompt_xxxxx')
        - parser.add_argument('--model_path', type=str, default=None)
        - parser.add_argument('--model_lora_path', type=str, default=None)
        """
        
        self.args = args
        assert args.model in ['llama3_8b', 'llama3_70b']

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir
        )
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=args.cache_dir
        )
        if args.model_lora_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                args.model_lora_path
            )
        self.model.eval()
    
    
    def generate(self, messages, **gen_kwargs):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device) # type: ignore

        outputs = self.model.generate(
            input_ids,
            # max_new_tokens=1024,
            eos_token_id=self.terminators,
            # do_sample=False
            **gen_kwargs
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        return result