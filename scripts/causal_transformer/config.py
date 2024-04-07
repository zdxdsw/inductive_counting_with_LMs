from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = 1234
    date = "debug"
    num_hidden_layers = 4
    vocab = []
    hidden_size = 1024
    num_attention_heads = 8
    mlp_dim_multipler = 4
    max_position_embeddings = 140
    layer_norm_epsilon = 1e-5
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = True
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    activation_function = 'relu'
    initializer_range = 0.02
    max_grad_norm = 0.3
    output_dir = "output"
    ckpt_dir = "/data/yingshac/llms_do_math/scripts/causal_transformer/output"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 1024
    eval_accumulation_steps = 1
    logging_steps = 100
    warmup_steps = 1000
    learning_rate = 1e-4
    num_epochs = 10
    save_every_steps = 20000
    eval_every_steps = 5000
    load_from_dir = None #"0406_134007" #
    init_from_ckpt = None


@dataclass
class counting_samesymbol_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol"

@dataclass
class counting_diffsymbol_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol"
