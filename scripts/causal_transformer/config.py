from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = 1234
    num_hidden_layers = 1
    vocab = []
    hidden_size = 1024
    num_attention_heads = 8
    mlp_dim_multipler = 4
    max_position_embeddings = 128
    layer_norm_epsilon = 1e-5
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = True
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    activation_function = 'silu'
    initializer_range = 0.02
    max_grad_norm = 0.3
    output_dir = "output"
    ckpt_dir = "/data/yingshac/llms_do_math/scripts/causal_transformer/output"
    per_device_train_batch_size = 128
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 512
    eval_accumulation_steps = 1
    logging_steps = 200
    warmup_steps = 200
    learning_rate = 1e-4
    num_train_epochs = 1
    load_from_dir = None


@dataclass
class counting_samesymbol_Config(Basic_Config):
    vocab = ['<pad>', 'a'] + [str(i) for i in range(101)]
    data_path = "../../data/rasp_primitives/counting_samesymbol"
