import torch, os, re, random, json, pytz
from datetime import datetime
timezone = pytz.timezone('America/New_York') 
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from trl import SFTTrainer
from transformers.modeling_utils import unwrap_model
from peft.utils import get_peft_model_state_dict
from accelerate import Accelerator

class NiceSFTTrainer(SFTTrainer):
    
    def save_model(
        self, 
        output_dir: Optional[str] = None, 
        _internal_call: bool = False
    ):
        if Accelerator().process_index: return

        if output_dir is None: 
            output_dir = self.args.output_dir

        # If we are executing this function, we are the process zero, so we don't check for that. <-- This is a lie (as least with accelerate launch)
        os.makedirs(output_dir, exist_ok=True)
        self.model.create_or_update_model_card(output_dir)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(
            unwrap_model(self.model), 
            adapter_name="default"
        )

        torch.save(output_state_dict, os.path.join(output_dir, "adapter_model.bin"))
        print(f"----- save model to {output_dir} -----")


def preprocess(prompt_template, input_str, answer=None, eos_token="</s>"):
  prompt = prompt_template.format(input_str)
  response = f"{str(answer) + '.' + eos_token if answer else ''} "
  text = "### Question: {}\n ### Answer: {}".format(prompt, response) #(" ").join([prompt, response])
  return text

def formatting_func(instance, prompt_template):
    #global prompt_template
    output = []
    for d, s in zip(instance["input_str"], instance["answer"]):
        op = preprocess(prompt_template, d, s)
        output.append(op)
    return output

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_preds, config, tokenizer, save_to, header=None):
    preds, labels = eval_preds
    em, count = 0, 0
    with open(os.path.join(config.output_dir, config.date, save_to, f'{datetime.now(timezone).strftime("%m%d_%H%M%S")}.txt'), "w") as f:
        if header: f.write(header+"\n")
        for pred, label in zip(preds, labels):
            # data collator would assign -100 to tokens who don't require loss calcupation
            p = 0
            while True:
                if label[p] != -100:
                    break
                p += 1
            
            a = [i for i in label[p:] if i != -100]
            o = [i for i in pred[p-1:] if i != -100]

            gth_response = tokenizer.decode(a, skip_special_tokens=True)
            pred_response = tokenizer.decode(o, skip_special_tokens=True)
            gth_tokenized = tokenizer.convert_ids_to_tokens(o)
            pred_tokenized = tokenizer.convert_ids_to_tokens(a)
            
            f.write(json.dumps([gth_response, pred_response, gth_tokenized, pred_tokenized])+"\n")
            
            try: gth = re.findall(r'([\d+-]+)', gth_response)[0]
            except IndexError: 
                print("a = ", a)
                print("gth_response = ", gth_response)
                continue
            
            pred = -1
            find = re.findall(r'([\d+-]+)', pred_response)
            if find: pred = find[0]
            
            em += int(gth == pred)
            count += 1
    print(f"eval acc = {round(em/count, 4)}")
    return {'accuracy': em/count}


