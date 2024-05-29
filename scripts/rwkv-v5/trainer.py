########################################################################################################
# Reference: The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging, os, json, sys
logging.basicConfig(level=logging.ERROR) #INFO) #logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


if __name__ == "__main__":
    from config import *
    sys.path.append("../")
    from causal_transformer.config_taskspecific import *
    from causal_transformer.utils import trim_task
    
    config = Basic_Config()

    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    parser = ArgumentParser()
    assert pl.__version__[0]=='1'
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--date', type=str, default="debug")
    parser.add_argument('--task', type=str)
    """ Will take care of these later """
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--my_testing", default='x060', type=str)

    args = parser.parse_args()
    
    config = eval(f"{args.task}_Config")()

    tmp = json.load(open(os.path.join(config.output_dir, args.date, "config.json"), "r"))
    for k, v in tmp.items():
        setattr(config, k, v)
    config.date = args.date


    parser.add_argument('--train_data_path', type=str, default=config.train_data_path)
    parser.add_argument('--eval_data_path', type=str, default=config.eval_data_path)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=config.per_device_eval_batch_size)
    parser.add_argument('--test_files', type=str, default=config.test_files)

    parser.add_argument("--logging_steps", default=config.logging_steps, type=int)
    parser.add_argument("--eval_every_steps", default=config.eval_every_steps, type=int)
    parser.add_argument("--proj_dir", default=os.path.join(config.output_dir, config.date), type=str)
    parser.add_argument("--ckpt_dir", default=os.path.join(config.ckpt_dir, config.date, "ckpts"), type=str)
    parser.add_argument("--random_seed", default=config.seed, type=int)
    parser.add_argument("--ctx_len", default=config.max_seq_len, type=int) # to be used in make_data.py
    parser.add_argument("--epoch_count", default=config.num_epochs, type=int) # = num_epochs # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_save", default=1, type=int)  # save the model every [epoch_save] "epochs"
    parser.add_argument("--micro_bsz", default=config.per_device_train_batch_size, type=int) # per_device_train_batch_size # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=config.num_hidden_layers, type=int)
    parser.add_argument("--n_embd", default=config.hidden_size, type=int)
    parser.add_argument("--lr_init", default=config.learning_rate, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--warmup_steps", default=config.warmup_steps, type=int)  # try 50 if you load a model
    parser.add_argument("--weight_decay", default=config.weight_decay, type=float) # try 0.1 / 0.01 / 0.001
    parser.add_argument("--dropout", default=config.dropout, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    
    
    """ Don't change, use default value according to the original repo """
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--ds_bucket_mb", default=2, type=int) # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage) # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--train_type", default="", type=str) # ""/"states"
    parser.add_argument("--weight_decay_final", default=-1, type=float) # for weight_decay schedule
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    
    

    args = parser.parse_args()

    ########################################################################################################

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from datasets import load_dataset, concatenate_datasets
    from rwkv_utils import sequences_collator, train_callback, generate_init_weight
    from functools import partial
    
    
    if "deepspeed" in args.strategy: import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0: seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = 1.0
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    args.lr_final = args.lr_init # disable lr decay
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(config.max_seq_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = args.train_type

    
    if not os.path.exists(args.proj_dir): os.makedirs(args.proj_dir)
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)

    rank_zero_info(
        f"""
        ############################################################################
        #
        # RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
        #
        # Data = <log data file here>, ProjDir = {args.proj_dir}
        #
        # Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (inclusive) , save every {args.epoch_save} epoch
        #
        # Model = {args.n_layer} n_layer, {args.n_embd} n_embd, <log max_seq_len here> ctx_len
        #
        # Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
        #
        # Found torch {torch.__version__}, recommend latest torch
        # Found deepspeed {deepspeed.__version__}, recommend latest deepspeed
        # Found pytorch_lightning {pl.__version__}, recommend 1.9.5
        #
        ############################################################################
        """
    )


    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    args.vocab = config.vocab
    args.vocab_size = len(args.vocab)

    # --------------------------------------------------------------------------
    tasks = [trim_task(args.task)] + config.aux_tasks

    train_data = concatenate_datasets(
                        [load_dataset(
                                "text", 
                                data_files={"train": f"{config.train_data_path}/{task}/train.txt"}
                                )['train'] for task in tasks]
                    )
    val_data = load_dataset(
                        "text", 
                        data_files={"validation": f"{config.eval_data_path}/{trim_task(args.task)}/val.txt"}
                        )['validation']
    
    args.max_seen_len = max([len([x for x in json.loads(l['text'])[0] if x != "<pad>"]) for l in val_data])
    print(f"max_seen_len for {config.task} = {args.max_seen_len}")
    print(f"\n-------------------------------\n------ #training = {len(train_data)} ------\n-------------------------------\n")

    collator = partial(sequences_collator, 
                        w2i={w:i for i,w in enumerate(args.vocab)}, 
                        max_seq_len=config.max_seq_len,
                    )

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.micro_bsz, collate_fn=collator, pin_memory=True, num_workers=1, persistent_workers=False)
    args.epoch_steps = len(train_dataloader)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=64, collate_fn=collator, pin_memory=True, num_workers=1, persistent_workers=False)

    val_dataloaders = {"val": val_dataloader}
    for ood_test_file in config.test_files:
        test_data = load_dataset("text",  data_files={ood_test_file: f"{config.eval_data_path}/{trim_task(args.task)}/{ood_test_file}.txt"})
        val_dataloaders[ood_test_file] = DataLoader(test_data[ood_test_file], shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator)


    # --------------------------------------------------------------------------

    from model import RWKV
    model = RWKV(args)

    if len(args.load_model) == 0: # or args.my_pile_stage == 1:  # shall we build the initial weights?
        rank_zero_info(f"\n\n----------------------- Building Initial Weights... -----------------------\n")
        init_weight_name = f"{args.ckpt_dir}/init_rwkv.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    
    load_dict = torch.load(args.load_model, map_location="cpu")
    load_keys = list(load_dict.keys())
    for k in load_keys:
        if k.startswith('_forward_module.'):
            load_dict[k.replace('_forward_module.','')] = load_dict[k]
            del load_dict[k]
    model.load_state_dict(load_dict)
    rank_zero_info(f"\n\n----------------------- Finish Loading {args.load_model}... -----------------------\n")


    rank_zero_info(f"\n\n----------------------- Create Trainer... -----------------------\n")

    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
    class MyTQDMProgressBar(TQDMProgressBar):
        def init_train_tqdm(self):
            return Tqdm(
                desc=self.train_description,
                position=(2 * self.process_position),
                disable=self.is_disabled,
                leave=True,
                dynamic_ncols=True,
                file=sys.stderr,
                smoothing=0,
                #bar_format=self.BAR_FORMAT,
            )

    args.max_epochs = args.epoch_count
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args), MyTQDMProgressBar(refresh_rate=args.logging_steps)],
    )

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

    print(f"\n\n\n----------------------- Start Training {config.date} -----------------------\n\n")
    trainer.fit(model, train_dataloader)

    print("\n\n\n----------------------- Start Inference -----------------------\n\n")
    infr_command = "python tester.py --handle {} --test_files \"{}\"".format(config.date, " ".join(["val"]+config.test_files))
    os.system(infr_command)
    print(f"Finish!!! {config.date}")