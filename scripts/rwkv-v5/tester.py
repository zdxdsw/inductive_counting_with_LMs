########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging, os, json, sys
logging.basicConfig(level=logging.INFO)

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

    parser.add_argument('--handle', type=str)
    parser.add_argument('--load_from_epochs', type=str, default="all") # str of space separated ints
    parser.add_argument('--test_files', type=str, default=None)
    parser.add_argument('--loop', type=int, default=1) # number of times to loop through the test_dataloader
    parser.add_argument("--my_testing", default='x060', type=str)

    args = parser.parse_args()
    
    config = Basic_Config()

    tmp = json.load(open(os.path.join(config.output_dir, args.handle, "config.json"), "r"))
    for k, v in tmp.items():
        setattr(config, k, v)


    parser.add_argument('--eval_data_path', type=str, default=config.eval_data_path)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=config.per_device_eval_batch_size)
    parser.add_argument("--proj_dir", default=os.path.join(config.output_dir, config.date), type=str)
    parser.add_argument("--ckpt_dir", default=os.path.join(config.ckpt_dir, config.date, "ckpts"), type=str)
    parser.add_argument("--random_seed", default=config.seed, type=int)
    parser.add_argument("--ctx_len", default=config.max_seq_len, type=int) # to be used in make_data.py
    parser.add_argument("--n_layer", default=config.num_hidden_layers, type=int)
    parser.add_argument("--n_embd", default=config.hidden_size, type=int)
    parser.add_argument("--dropout", default=config.dropout, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--precision", default="bf16", type=str) 
    parser.add_argument("--devices", default=1, type=int)  # number of GPUs
    parser.add_argument('--accelerator', type=str, default="gpu")
    
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

    import os, warnings, math, datetime, sys, time, pytz
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from datasets import load_dataset, concatenate_datasets
    from rwkv_utils import sequences_collator, train_callback, generate_init_weight
    from functools import partial
    from datetime import datetime
    timezone = pytz.timezone('America/Los_Angeles') 
    
    #if "deepspeed" in args.strategy: import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0: seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    args.my_timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    #args.gradient_clip_val = 1.0
    #args.num_sanity_val_steps = 0
    #args.check_val_every_n_epoch = int(1e20)
    #args.log_every_n_steps = int(1e20)
    #args.max_epochs = -1  # continue forever
    #args.betas = (args.beta1, args.beta2)
    #args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    #args.lr_final = args.lr_init # disable lr decay
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(config.max_seq_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = args.train_type


    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    #if "deepspeed_stage_3" in args.strategy:
    #    os.environ["RWKV_JIT_ON"] = "0"

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

    #vocab = [str(i) for i in range(101)] + ['<pad>', 'a', '<b>']
    args.vocab = config.vocab
    args.vocab_size = len(args.vocab)

    # --------------------------------------------------------------------------
    val_file = open(f"{config.eval_data_path}/{trim_task(config.task)}/val.txt", "r").readlines()
    args.max_seen_len = max([len([x for x in json.loads(l)[0] if x != "<pad>"]) for l in val_file])
    print(f"\nmax_seen_len for {config.task} = {args.max_seen_len}\n")

    collator = partial(sequences_collator, 
                        w2i={w:i for i,w in enumerate(args.vocab)}, 
                        max_seq_len=config.max_seq_len,
                    )

    test_dataloaders = {}
    for test_file in args.test_files.split():
        test_data = load_dataset("text",  data_files={test_file: f"{config.eval_data_path}/{trim_task(config.task)}/{test_file}.txt"})
        test_dataloaders[test_file] = DataLoader(test_data[test_file], shuffle=False, batch_size=config.per_device_eval_batch_size, collate_fn=collator)

    
    # --------------------------------------------------------------------------
    
    device = "cuda"
    data_path = f"{config.eval_data_path}/{trim_task(config.task)}"
    avail_ckpts = [x for x in os.listdir(args.ckpt_dir) if "init" not in x]
    avail_ckpts = sorted(avail_ckpts, key=lambda x: int(x.split("_")[0]))
    if args.load_from_epochs != "all":
        avail_ckpts = [ckpt for ckpt in avail_ckpts if int(ckpt.split("_")[0]) in [int(e) for e in args.load_from_epochs.split()]]


    from model import RWKV
    model = RWKV(args)

    for load_from_pt in avail_ckpts:
        load_weight_name = f"{args.ckpt_dir}/{load_from_pt}"    
        load_dict = torch.load(load_weight_name, map_location="cpu")
        model.load_state_dict(load_dict)
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()

        last_date = ""
        for split, test_dataloader in test_dataloaders.items():

            counting_correct, counting_demo, last_correct, last_demo, unseen_len_correct, unseen_len_demo, correct, demo = 0, 0, 0, 0, 0, 0, 0, 0
            test_losses = []
            
            testing_output = {}

            _date = datetime.now(timezone).strftime("%m%d_%H%M%S") + "_" + load_from_pt.split("_")[0]
            if _date == last_date: 
                time.sleep(1)
                _date = datetime.now(timezone).strftime("%m%d_%H%M%S") + "_" + load_from_pt.split("_")[0]
            last_date = _date

            k = 0
            for loop in range(args.loop):
                for i, batch in enumerate(test_dataloader):
                    (loss, 
                        logits, 
                        _counting_correct, 
                        _counting_demo, 
                        _last_correct, 
                        _last_demo, 
                        _unseen_len_correct,
                    _unseen_len_demo) = model.predict_step(batch, i)

                    test_losses.append(loss)

                    counting_correct += _counting_correct
                    counting_demo += _counting_demo
                    last_correct += _last_correct
                    last_demo += _last_demo
                    unseen_len_correct += _unseen_len_correct
                    unseen_len_demo += _unseen_len_demo
                    correct += (_counting_correct + _last_correct)
                    demo += (_counting_demo + _last_demo)

                    #print(logits.shape)
                    for input_id, gth_id, pred_id in zip(batch['input_id'], batch['label'], logits.argmax(dim=-1)):
                        #print(input_id)
                        input_seq = [config.vocab[i] for i in input_id if config.vocab[i]!='<pad>']
                        gth_seq = [config.vocab[gth_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1]
                        pred_seq = [config.vocab[pred_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1][:len(gth_seq)]
                        testing_output[k] = {
                            "input": " ".join(input_seq),
                            "gth": " ".join(gth_seq),
                            "pred": " ".join(pred_seq),
                        }
                        k+=1
                
            print(f""" {split} acc, load from {load_from_pt}
                    | Test Loss: {round(np.mean(test_losses), 4)} 
                    | Test Acc: {round(correct/demo, 4)} 
                    | Test Counting Acc: {round(counting_correct/counting_demo, 4)} 
                    | Test Last Acc: {round(last_correct/last_demo, 4)}
                    | Test Unseen Len Acc: {round(unseen_len_correct/unseen_len_demo, 4) if unseen_len_demo != 0 else -1}
                """)

            save_dir = "test_samples" if "test" in split else "val_samples"
            os.makedirs(f"{config.output_dir}/{args.handle}/{save_dir}", exist_ok=True)
            json.dump({
                    "test_data_file":  f"{data_path}/{split}.txt",
                    "load_from": f"{args.handle}/{load_from_pt}",
                    "test_acc": round(correct/demo, 4),
                    "test_counting_acc": round(counting_correct/counting_demo, 4),
                    "test_last_acc": round(last_correct/last_demo, 4),
                    "test_unseen_len_acc": round(unseen_len_correct/unseen_len_demo, 4) if unseen_len_demo != 0 else -1,
                    "test_loss": round(np.mean(test_losses), 4),
                    "testing_output": testing_output,
                }, 
                open(f"{config.output_dir}/{args.handle}/{save_dir}/{_date}.json", "w"), indent=2)
        



    

    