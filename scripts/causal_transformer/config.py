from dataclasses import dataclass

@dataclass
class Basic_Config:
    seed = 1234
    date = "debug"
    num_hidden_layers = 1
    vocab = []
    task = ""
    aux_tasks = []
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
    tie_word_embeddings = True
    activation_function = 'relu'
    initializer_range = 0.02
    max_grad_norm = 0.3
    output_dir = "output"
    ckpt_dir = "/data/yingshac/llms_do_math/scripts/causal_transformer/output"
    train_data_path = "/data/yingshac/llms_do_math/data/rasp_primitives/"
    eval_data_path = "../../data/rasp_primitives/"
    test_files = ["ood_test"]
    per_device_train_batch_size = 8
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
    rotary_posemb = False
    rotary_posemb_shift = False
    rotary_posemb_rdmz = False
    load_from_dir = None #"0423_105639" # 
    init_from_ckpt = None

@dataclass
class debug_Config(Basic_Config):
    vocab = ['<pad>', 'a', 'b', '1', '2', '3']

@dataclass
class counting_samesymbol_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']

@dataclass
class counting_diffsymbol_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')

@dataclass
class counting_samesymbol_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addbigram']
    test_files = ["ood_test", "addbigram_test"]

@dataclass
class counting_diffsymbol_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a'] + list('abcdefghijklmnopqrstuvwxyz')
    aux_tasks = ['addbigram']
    test_files = ["ood_test", "addbigram_test"]

@dataclass
class counting_samesymbol_shiftedstart_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']

@dataclass
class counting_diffsymbol_shiftedstart_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a'] + list('abcdefghijklmnopqrstuvwxyz')

@dataclass
class counting_samesymbol_shiftedstart_addtable_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addtable']
    test_files = ["ood_test", "addtable_test"]

@dataclass
class counting_samesymbol_blankhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<blk>']
    max_position_embeddings = 256

@dataclass
class counting_samesymbol_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_position_embeddings = 256

@dataclass
class counting_samesymbol_padhelper_addtable_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_position_embeddings = 256
    aux_tasks = ['addtable_padhelper']
    test_files = ["ood_test", "addtable_padhelper_test"]

@dataclass
class counting_samesymbol_mod10_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>', 'a']

@dataclass
class counting_diffsymbol_mod10_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')

@dataclass
class counting_samesymbol_mod10_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>', 'a']
    max_position_embeddings = 210

@dataclass
class counting_diffsymbol_mod10_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 210

@dataclass
class counting_samesymbol_mod10_100_200_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>', 'a']
    max_position_embeddings = 256

@dataclass
class counting_diffsymbol_mod10_100_200_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 256

@dataclass
class counting_samesymbol_mod11_Config(Basic_Config):
    vocab = [str(i) for i in range(12)] + ['<pad>', 'a']

@dataclass
class counting_diffsymbol_mod11_Config(Basic_Config):
    vocab = [str(i) for i in range(12)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')

@dataclass
class counting_samesymbol_mod13_Config(Basic_Config):
    vocab = [str(i) for i in range(14)] + ['<pad>', 'a']
    max_position_embeddings = 140

@dataclass
class counting_diffsymbol_mod13_Config(Basic_Config):
    vocab = [str(i) for i in range(14)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 140

@dataclass
class counting_samesymbol_mod14_Config(Basic_Config):
    vocab = [str(i) for i in range(15)] + ['<pad>', 'a']
    max_position_embeddings = 150

@dataclass
class counting_diffsymbol_mod14_Config(Basic_Config):
    vocab = [str(i) for i in range(15)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 150

@dataclass
class counting_samesymbol_mod15_Config(Basic_Config):
    vocab = [str(i) for i in range(16)] + ['<pad>', 'a']
    max_position_embeddings = 160

@dataclass
class counting_diffsymbol_mod15_Config(Basic_Config):
    vocab = [str(i) for i in range(16)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 160

@dataclass
class counting_samesymbol_mod16_Config(Basic_Config):
    vocab = [str(i) for i in range(17)] + ['<pad>', 'a']
    max_position_embeddings = 180

@dataclass
class counting_diffsymbol_mod16_Config(Basic_Config):
    vocab = [str(i) for i in range(17)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 180

@dataclass
class counting_samesymbol_mod17_Config(Basic_Config):
    vocab = [str(i) for i in range(18)] + ['<pad>', 'a']
    max_position_embeddings = 180

@dataclass
class counting_diffsymbol_mod17_Config(Basic_Config):
    vocab = [str(i) for i in range(18)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 180

@dataclass
class counting_samesymbol_mod18_Config(Basic_Config):
    vocab = [str(i) for i in range(19)] + ['<pad>', 'a']
    max_position_embeddings = 190

@dataclass
class counting_diffsymbol_mod18_Config(Basic_Config):
    vocab = [str(i) for i in range(19)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 190

@dataclass
class counting_samesymbol_mod19_Config(Basic_Config):
    vocab = [str(i) for i in range(20)] + ['<pad>', 'a']
    max_position_embeddings = 200

@dataclass
class counting_diffsymbol_mod19_Config(Basic_Config):
    vocab = [str(i) for i in range(20)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 200

@dataclass
class counting_samesymbol_mod20_Config(Basic_Config):
    vocab = [str(i) for i in range(21)] + ['<pad>', 'a']
    max_position_embeddings = 210

@dataclass
class counting_diffsymbol_mod20_Config(Basic_Config):
    vocab = [str(i) for i in range(21)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 210

@dataclass
class counting_raspL_Config(Basic_Config):
    vocab = [str(i) for i in range(156)] + ['<pad>', '<', '<eos>', '<sos>'] 
    max_position_embeddings = 256

@dataclass
class counting_selective_Config(Basic_Config):
    vocab = [str(i) for i in range(51)] + ['<pad>', 'a', 'b'] 

@dataclass
class counting_selective_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(51)] + ['<pad>', 'a', 'b'] 
    max_position_embeddings = 256

