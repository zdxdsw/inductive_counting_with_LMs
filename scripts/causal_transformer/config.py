from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = 1234
    date = "debug"
    hf_cache_dir = '/data/yingshac/hf_cache'
    num_hidden_layers = 2
    vocab = []
    task = ""
    aux_tasks = []
    hidden_size = 1024
    num_attention_heads = 8
    mlp_dim_multipler = 4
    max_position_embeddings = 128
    max_seq_len = 128
    freeze_null_emb = False
    layer_norm_epsilon = 1e-5
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = True
    resid_pdrop = 0. #0.1
    embd_pdrop = 0. #0.1
    attn_pdrop = 0. #0.1
    must_attend_to_identity = False
    tie_word_embeddings = False
    activation_function = 'relu'
    initializer_range = 0.02
    max_grad_norm = 0.3
    output_dir = "output"
    ckpt_dir = "/data/yingshac/llms_do_math/scripts/causal_transformer/output"
    train_data_path = "/data/yingshac/llms_do_math/data/rasp_primitives/"
    eval_data_path = "../../data/rasp_primitives/"
    test_files = ["ood_test"]
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 64
    eval_accumulation_steps = 1
    logging_steps = 100
    warmup_steps = 3000
    learning_rate = 1e-4
    num_epochs = 10
    #save_every_steps = 20000
    eval_every_steps = 40000
    absolute_posemb = False
    absolute_posemb_shift = False
    absolute_posemb_rdmz = False
    rotary_posemb = True
    rotary_posemb_shift = True
    rotary_posemb_rdmz = False
    scaler_posemb = False
    scaler_posemb_shift = False
    sinusoidal_posemb = False
    sinusoidal_posemb_shift = False
    load_from_dir = None #"0527_162257" # 
    init_from_ckpt = None


@dataclass
class Default_Config:
    seed = 1234
    hf_cache_dir = None
    must_attend_to_identity = False
    tie_word_embeddings = False
    absolute_posemb = False
    absolute_posemb_shift = False
    absolute_posemb_rdmz = False
    rotary_posemb = False
    rotary_posemb_shift = False
    rotary_posemb_rdmz = False
    scaler_posemb = False
    scaler_posemb_shift = False
    sinusoidal_posemb = False
    sinusoidal_posemb_shift = False
    freeze_null_emb = False
    aux_tasks = []
    max_seq_len = 128