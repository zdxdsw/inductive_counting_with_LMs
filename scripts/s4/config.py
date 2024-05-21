from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = [1234, 42, 319]
    model = "RNN"
    date = "debug"
    num_hidden_layers = 1
    vocab = []
    task = ""
    aux_tasks = []
    hidden_size = 1024
    max_seq_len = 128
    freeze_null_emb = False
    dropout = 0.0
    tie_word_embeddings = False
    initializer_range = 0.02
    output_dir = "output"
    ckpt_dir = "/home/ubuntu/largefiles/llms_do_math/scripts/causal_transformer/output"
    train_data_path = "/home/ubuntu/largefiles/llms_do_math/data/rasp_primitives/"
    eval_data_path = "../../data/rasp_primitives/"
    test_files = ["ood_test"]
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 64
    eval_accumulation_steps = 1
    logging_steps = 500
    #warmup_steps = 0 #3000
    learning_rate = 0.001
    weight_decay = 0.01
    num_epochs = 50
    eval_every_steps = 10000
    load_from_dir = None # "0514_183327" #
    init_from_ckpt = None
    rnn_actv = "tanh"


@dataclass
class Default_Config:
    seed = 1234
    tie_word_embeddings = False
    initializer_range = 0.02
    model = "S4"
    freeze_null_emb = False
    rnn_actv = "tanh"