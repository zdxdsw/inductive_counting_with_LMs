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
    max_position_embeddings = 128
    layer_norm_epsilon = 1e-5
    scale_attn_weights = True
    scale_attn_by_inverse_layer_idx = True
    resid_pdrop = 0. #0.1
    embd_pdrop = 0. #0.1
    attn_pdrop = 0. #0.1
    must_attend_to_identity = False
    activation_function = 'relu'
    initializer_range = 0.02
    max_grad_norm = 0.3
    output_dir = "output"
    ckpt_dir = "/data/yingshac/llms_do_math/scripts/causal_transformer/output"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 1
    per_device_eval_batch_size = 128
    eval_accumulation_steps = 1
    logging_steps = 100
    warmup_steps = 3000
    learning_rate = 1e-4
    num_epochs = 10
    #save_every_steps = 20000
    eval_every_steps = 10000
    absolute_posemb = True
    absolute_posemb_shift = False
    absolute_posemb_rdmz = True
    rotary_posemb = False
    rotary_posemb_shift = False
    rotary_posemb_rdmz = False
    load_from_dir = None #"0410_142755" #
    init_from_ckpt = None


@dataclass
class counting_samesymbol_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol"

@dataclass
class counting_diffsymbol_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol"

@dataclass
class counting_samesymbol_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol_addbigram"

@dataclass
class counting_diffsymbol_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol_addbigram"

@dataclass
class counting_samesymbol_shiftedstart_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol_shiftedstart"

@dataclass
class counting_diffsymbol_shiftedstart_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol_shiftedstart"

@dataclass
class counting_samesymbol_blankhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<blk>']
    data_path = "../../data/rasp_primitives/counting_samesymbol_blankhelper"
    max_position_embeddings = 256

@dataclass
class counting_samesymbol_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol_padhelper"
    max_position_embeddings = 256

@dataclass
class counting_samesymbol_mod10_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol_mod10"

@dataclass
class counting_diffsymbol_mod10_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol_mod10"

@dataclass
class counting_diffsymbol_mod10_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol_mod10_padhelper"
    max_position_embeddings = 256

@dataclass
class counting_samesymbol_mod20_Config(Basic_Config):
    vocab = [str(i) for i in range(21)] + ['<pad>', 'a']
    data_path = "../../data/rasp_primitives/counting_samesymbol_mod20"
    max_position_embeddings = 256

@dataclass
class counting_diffsymbol_mod20_Config(Basic_Config):
    vocab = [str(i) for i in range(21)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    data_path = "../../data/rasp_primitives/counting_diffsymbol_mod20"
    max_position_embeddings = 256

@dataclass
class counting_raspL_Config(Basic_Config):
    vocab = [str(i) for i in range(156)] + ['<pad>', '<', '<eos>', '<sos>'] 
    data_path = "../../data/rasp_primitives/counting_raspL"
    max_position_embeddings = 256

@dataclass
class counting_selective_Config(Basic_Config):
    vocab = [str(i) for i in range(51)] + ['<pad>', 'a', 'b'] 
    data_path = "../../data/rasp_primitives/counting_selective"

@dataclass
class counting_selective_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(51)] + ['<pad>', 'a', 'b'] 
    data_path = "../../data/rasp_primitives/counting_selective_padhelper"
    max_position_embeddings = 256

