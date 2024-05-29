from dataclasses import dataclass
from config import Basic_Config

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
class counting_samesymbol_plain_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addbigram']
    test_files = ["ood_test", "addbigram_test"]

@dataclass
class counting_samesymbol_plain2_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addbigram']
    test_files = ["ood_test", "addbigram_test"]

@dataclass
class counting_samesymbol_plain2_addbigram_blkseq_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<b>']
    aux_tasks = ['addbigram_blkseq']
    test_files = ["ood_test", "addbigram_blkseq_test"]

@dataclass
class counting_samesymbol_plain2_addbigram_nullseq_Config(Basic_Config):
    vocab = ['<null>'] + [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addbigram_nullseq']
    test_files = ["ood_test", "addbigram_nullseq_test"]
    freeze_null_emb = True

@dataclass
class counting_samesymbol_plain3_addbigram_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<b>']
    aux_tasks = ['addbigram']
    test_files = ["ood_test", "addbigram_test"]

@dataclass
class counting_samesymbol_plain3_addbigram_blkseq_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<b>']
    aux_tasks = ['addbigram_blkseq']
    test_files = ["ood_test", "addbigram_blkseq_test"]

@dataclass
class counting_samesymbol_plain3_addbigram_nullseq_Config(Basic_Config):
    vocab = ['<null>'] + [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addbigram_nullseq']
    test_files = ["ood_test", "addbigram_nullseq_test"]
    freeze_null_emb = True

@dataclass
class counting_samesymbol_shiftedstart4_addbigram_blkseq_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<b>']
    aux_tasks = ['addbigram_shiftedblkseq']
    test_files = ["ood_test", "addbigram_shiftedblkseq_test"]

@dataclass
class counting_samesymbol_shiftedstart4_addbigram_nullseq_Config(Basic_Config):
    vocab = ['<null>'] + [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addbigram_shiftednullseq']
    test_files = ["ood_test", "addbigram_shiftednullseq_test"]
    freeze_null_emb = True

@dataclass
class counting_samesymbol_blankhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<blk>']
    max_seq_len = 210

@dataclass
class counting_samesymbol_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_seq_len = 210

@dataclass
class counting_samesymbol_padhelper_addtable_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_seq_len = 210
    aux_tasks = ['addtable_padhelper']
    test_files = ["ood_test", "addtable_padhelper_test"]

@dataclass
class counting_samesymbol_shiftedstart_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']

@dataclass
class counting_samesymbol_shiftedstart2_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']

@dataclass
class counting_samesymbol_shiftedstart3_Config(Basic_Config):
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
class counting_samesymbol_shiftedstart2_addtable_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addtable']
    test_files = ["ood_test", "addtable_test"]

@dataclass
class counting_samesymbol_shiftedstart3_addtablelarge_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    aux_tasks = ['addtable_large']
    test_files = ["ood_test", "addtable_test"]

@dataclass
class counting_samesymbol_shiftedstart2_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_seq_len = 210

@dataclass
class counting_samesymbol_shiftedstart3_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_seq_len = 210

@dataclass
class counting_samesymbol_shiftedstart2_padhelper_addtable_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_seq_len = 210
    aux_tasks = ['addtable_padhelper']
    test_files = ["ood_test", "addtable_padhelper_test"]

@dataclass
class counting_samesymbol_shiftedstart3_padhelper_addtablelarge_Config(Basic_Config):
    vocab = [str(i) for i in range(101)] + ['<pad>', 'a']
    max_seq_len = 210
    aux_tasks = ['addtable_padhelper_large']
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
    max_seq_len = 210

@dataclass
class counting_diffsymbol_mod10_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_seq_len = 210

@dataclass
class counting_samesymbol_mod10_100_200_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>', 'a']
    max_position_embeddings = 256
    max_seq_len = 256

@dataclass
class counting_diffsymbol_mod10_100_200_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 256
    max_seq_len = 256

@dataclass
class counting_samesymbol_mod10bos_Config(Basic_Config):
    vocab = [str(i) for i in range(11)] + ['<pad>', 'a']

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
    max_seq_len = 140

@dataclass
class counting_diffsymbol_mod13_Config(Basic_Config):
    vocab = [str(i) for i in range(14)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 140
    max_seq_len = 140

@dataclass
class counting_samesymbol_mod14_Config(Basic_Config):
    vocab = [str(i) for i in range(15)] + ['<pad>', 'a']
    max_position_embeddings = 150
    max_seq_len = 150

@dataclass
class counting_diffsymbol_mod14_Config(Basic_Config):
    vocab = [str(i) for i in range(15)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 150
    max_seq_len = 150

@dataclass
class counting_samesymbol_mod15_Config(Basic_Config):
    vocab = [str(i) for i in range(16)] + ['<pad>', 'a']
    max_position_embeddings = 160
    max_seq_len = 160

@dataclass
class counting_diffsymbol_mod15_Config(Basic_Config):
    vocab = [str(i) for i in range(16)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 160
    max_seq_len = 160

@dataclass
class counting_samesymbol_mod16_Config(Basic_Config):
    vocab = [str(i) for i in range(17)] + ['<pad>', 'a']
    max_position_embeddings = 180
    max_seq_len = 180

@dataclass
class counting_diffsymbol_mod16_Config(Basic_Config):
    vocab = [str(i) for i in range(17)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 180
    max_seq_len = 180

@dataclass
class counting_samesymbol_mod17_Config(Basic_Config):
    vocab = [str(i) for i in range(18)] + ['<pad>', 'a']
    max_position_embeddings = 180
    max_seq_len = 180

@dataclass
class counting_diffsymbol_mod17_Config(Basic_Config):
    vocab = [str(i) for i in range(18)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 180
    max_seq_len = 180

@dataclass
class counting_samesymbol_mod18_Config(Basic_Config):
    vocab = [str(i) for i in range(19)] + ['<pad>', 'a']
    max_position_embeddings = 190
    max_seq_len = 190

@dataclass
class counting_diffsymbol_mod18_Config(Basic_Config):
    vocab = [str(i) for i in range(19)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 190
    max_seq_len = 190

@dataclass
class counting_samesymbol_mod19_Config(Basic_Config):
    vocab = [str(i) for i in range(20)] + ['<pad>', 'a']
    max_position_embeddings = 200
    max_seq_len = 200

@dataclass
class counting_diffsymbol_mod19_Config(Basic_Config):
    vocab = [str(i) for i in range(20)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 200
    max_seq_len = 200

@dataclass
class counting_samesymbol_mod20_Config(Basic_Config):
    vocab = [str(i) for i in range(21)] + ['<pad>', 'a']
    max_position_embeddings = 210
    max_seq_len = 210

@dataclass
class counting_diffsymbol_mod20_Config(Basic_Config):
    vocab = [str(i) for i in range(21)] + ['<pad>'] + list('abcdefghijklmnopqrstuvwxyz')
    max_position_embeddings = 210
    max_seq_len = 210

@dataclass
class counting_raspL_Config(Basic_Config):
    vocab = [str(i) for i in range(156)] + ['<pad>', '<', '<eos>', '<sos>'] 
    max_position_embeddings = 256
    max_seq_len = 256

@dataclass
class counting_selective_Config(Basic_Config):
    vocab = [str(i) for i in range(51)] + ['<pad>', 'a', 'b'] 

@dataclass
class counting_selective_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(51)] + ['<pad>', 'a', 'b'] 
    max_seq_len = 210

@dataclass
class counting_selective_10items_Config(Basic_Config):
    vocab = [str(i) for i in range(10)] + ['<pad>'] 

@dataclass
class counting_selective_10items_padhelper_Config(Basic_Config):
    vocab = [str(i) for i in range(10)] + ['<pad>'] 
    max_seq_len = 210

@dataclass
class counting_selective_10items2_Config(Basic_Config):
    vocab = [str(i) for i in range(10)] + ['<pad>'] 

@dataclass
class counting_selective_10items_padhelper2_Config(Basic_Config):
    vocab = [str(i) for i in range(10)] + ['<pad>'] 
    max_seq_len = 210

@dataclass
class counting_selective_modular_4items_Config(Basic_Config):
    vocab = [str(i) for i in range(4)] + ['<pad>'] 
    max_position_embeddings = 130
    max_seq_len = 130