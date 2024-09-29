from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = [123, 42, 319]
    date = "debug"
    num_hidden_layers = 4
    vocab = []
    task = ""
    aux_tasks = []
    hidden_size = 128
    max_seq_len = 128
    dropout = 0.1
    output_dir = "output"
    ckpt_dir = "/data/tir/projects/tir7/user_data/yingshac/llms_do_math/scripts/rwkv/output"
    train_data_path = "/data/tir/projects/tir7/user_data/yingshac/llms_do_math/data/rasp_primitives/"
    eval_data_path = "../../data/rasp_primitives/"
    test_files = ["ood_test"]
    per_device_train_batch_size = 32
    per_device_eval_batch_size = 64
    eval_accumulation_steps = 1
    logging_steps = 100
    warmup_steps = 0
    learning_rate = 0.0001
    weight_decay = 0 #0.01
    num_epochs = 50
    eval_every_steps = 10000
    load_from_dir = None #"0507_205708" # 
    init_from_ckpt = None


@dataclass
class Default_Config:
    seed = 1234