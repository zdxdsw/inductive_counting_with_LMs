from dataclasses import dataclass

@dataclass
class FinetuningConfig:
    seed = 1234
    model_name = "meta-llama/Llama-2-7b-hf"
    output_dir = "output"
    ckpt_dir = "/data/yingshac/llms_do_math/scripts/llama/output"
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 2
    per_device_eval_batch_size = 16
    eval_accumulation_steps = 4
    optim = "paged_adamw_32bit"
    warmup_steps =200
    logging_steps = 200
    learning_rate = 1e-4
    num_train_epochs = 100
    max_grad_norm = 0.3
    max_seq_length = 160
    lora_target_modules = ["q_proj","k_proj","v_proj","o_proj"]
    bits = None
    date = "debug"
    zeroshot_eval = True
    load_from_dir = None #"0402_173505" # 
    
@dataclass
class FinetuningConfig_len(FinetuningConfig):
    prompt_path = "../../data/feed_decoder_LM/regular/len/q_prompt.txt"
    data_dir = "../../data/finetune/len/uniform_split"
    max_seq_length = 160

@dataclass
class FinetuningConfig_mode(FinetuningConfig):
    prompt_path = "../../data/feed_decoder_LM/regular/mode/q_prompt.txt"
    data_dir = [
        "../../data/finetune/mode/random_labels",
        #"../../data/finetune/mode/uniform_hard+_6_64_split",
    ]
    max_seq_length = 160

@dataclass
class FinetuningConfig_mode_2nd(FinetuningConfig):
    prompt_path = "../../data/feed_decoder_LM/regular/mode_2nd/q_prompt.txt"
    data_dir = [
        "../../data/finetune/mode_2nd/uniform_split",
        #"../../data/finetune/mode_2nd/uniform_hard+_split",
    ]
    max_seq_length = 160

@dataclass
class FinetuningConfig_parity(FinetuningConfig):
    prompt_path = "../../data/feed_decoder_LM/regular/parity/q_prompt.txt"
    data_dir = [
        "../../data/finetune/parity/random_labels",
    ]
    max_seq_length = 180
    #logging_steps = 2000
    #learning_rate = 1e-5

@dataclass
class FinetuningConfig_add(FinetuningConfig):
    prompt_path = "../../data/feed_decoder_LM/regular/add/q_prompt.txt"
    data_dir = "../../data/finetune/add/random_labels"
    #logging_steps = 1000
    max_seq_length = 410
    #per_device_train_batch_size = 2
    #gradient_accumulation_steps = 2
    #per_device_eval_batch_size = 16

@dataclass
class FinetuningConfig_sort(FinetuningConfig):
    prompt_path = "../../data/feed_decoder_LM/regular/sort/q_prompt.txt"
    data_dir = "../../data/finetune/sort/random_labels"
    logging_steps = 1000
    max_seq_length = 300
    #per_device_train_batch_size = 2
    #gradient_accumulation_steps = 2
    #per_device_eval_batch_size = 16