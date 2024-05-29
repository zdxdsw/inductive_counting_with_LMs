from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = [1234, 12, 123]
    date = "debug"
    model = "mamba"
    num_hidden_layers = 2
    vocab = []
    task = ""
    aux_tasks = []
    hidden_size = 128
    max_seq_len = 128
    output_dir = "output"
    ckpt_dir = "/home/ubuntu/largefiles/llms_do_math/scripts/mamba/output"
    train_data_path = "/home/ubuntu/largefiles/llms_do_math/data/rasp_primitives/"
    eval_data_path = "../../data/rasp_primitives/"
    test_files = ["ood_test"]
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 64
    eval_accumulation_steps = 1
    logging_steps = 500
    warmup_steps = 0 #3000
    learning_rate = 1e-4
    weight_decay = 0# 0.01
    num_epochs = 50
    eval_every_steps = 10000
    load_from_dir = None # "0520_081118" #
    init_from_ckpt = None
    precision = "fp32"


@dataclass
class Default_Config:
    seed = 1234
    precision = "fp16"
