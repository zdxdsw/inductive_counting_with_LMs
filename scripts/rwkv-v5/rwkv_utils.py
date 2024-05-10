import json, time, random, os
#import numpy as np
import torch
#from torch.nn import functional as F

# --------------------------------- moved from trainer.py from the original repo -------------------------------------
import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

def my_save(args, trainer, dd, ff):
    if '14b-run1' in ff:
        fn = ff.split('/')[-1]
        fff = '/dev/shm/' + fn
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-14b-4k/{fn} --quiet", shell=True)
    elif ('world/14b' in ff) or ('world/7b' in ff):
        aa = ff.split('/')[1]
        fn = ff.split('/')[-1]
        fff = f'/dev/shm/{aa}-{fn}'
        torch.save(dd, fff)
        subprocess.Popen(f" aws s3 mv {fff} s3://rwkv-world/{aa}-{fn} --quiet", shell=True)
    else:
        if 'deepspeed_stage_3' in args.strategy:
            trainer.save_checkpoint(ff, weights_only=True)
        else:
            torch.save(dd, ff)

class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
        # if args.cuda_cleanup > 0:
        #     torch.cuda.empty_cache()
        #real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule (disable decay)
        w_step = args.warmup_steps
        #if args.lr_final == args.lr_init or args.epoch_count == 0:
        lr = args.lr_init
        # else:
        #     decay_step = real_step
        #     decay_total = (args.epoch_count) * args.epoch_steps
        #     progress = (decay_step - w_step + 1) / (decay_total - w_step)
        #     progress = min(1, max(0, progress))

        #     if args.lr_final == 0 or args.lr_init == 0:  # linear decay
        #         lr = args.lr_init + (args.lr_final - args.lr_init) * progress
        #     else:  # exp decay
        #         lr = args.lr_init * math.exp(math.log(args.lr_final / args.lr_init) * pow(progress, 1))
        #     # if trainer.is_global_zero:
            #     print(trainer.global_step, decay_step, decay_total, w_step, progress, lr)

        # if args.my_exit_tokens != 0: # cosine decay
        #     real_tokens = real_step * args.ctx_len * args.real_bsz
        #     warmup_tokens = w_step * args.ctx_len * args.real_bsz
        #     progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
        #     progress = max(0, min(1, progress))
        #     lr_final_factor = args.lr_final / args.lr_init                
        #     lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(math.pi * progress)
        #     if args.my_exit_tokens > 0:
        #         lr = args.lr_init * lr_mult
        #     else:
        #         lr = (lr + args.lr_init * lr_mult) / 2
        #     if progress >= 1:
        #         if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):
        #             my_save(
        #                 args, trainer,
        #                 pl_module.state_dict(),
        #                 f"{args.proj_dir}/rwkv-final.pth",
        #             )
        #             exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.2 + 0.8 * trainer.global_step / w_step)

        if args.weight_decay_final > 0:
            wd_now = args.weight_decay * math.exp(math.log(args.weight_decay_final / args.weight_decay) * progress)
        else:
            wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            if args.layerwise_lr > 0:
                param_group["lr"] = lr * param_group["my_lr_scale"]
                # print(param_group["lr"], param_group["my_lr_scale"])
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now
        # rank_zero_info(f"{real_step} {lr}")

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
                print("\n\n----------------------- Trainer Strategy: -----------------------\n")
                print(f"{trainer.strategy.config}\n\n")
                trainer.my_log.write(f"{trainer.strategy.config}\n")
                trainer.my_log.flush()
                # if len(args.wandb) > 0:
                #     print("Login to wandb...")
                #     import wandb
                #     wandb.init(
                #         project=args.wandb,
                #         name=args.run_name + " " + args.my_timestamp,
                #         config=args,
                #         save_code=False,
                #     )
                #     trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        if trainer.is_global_zero and real_step % args.logging_steps == 0:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            if pl.__version__[0]=='2':
                trainer.my_loss = outputs["loss"]
            else:
                trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            self.log("s", float(real_step), prog_bar=True, on_step=True)

            # if len(args.wandb) > 0:
            #     lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
            #     if kt_s > 0:
            #         lll["kt/s"] = kt_s
            #     trainer.my_wandb.log(lll, step=int(real_step))
        # Disable save_model in the middle of an epoch
        # if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy): # save pth
        #     if args.magic_prime > 0:
        #         expand_factor = 1
        #         if int(real_step) == int(args.magic_prime * expand_factor // args.real_bsz) - 1 + int(args.my_random_steps):
        #             to_save_dict = pl_module.state_dict()
        #             my_save(
        #                 args, trainer,
        #                 to_save_dict,
        #                 f"{args.proj_dir}/rwkv-final.pth",
        #             )
                

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        # if pl.__version__[0]=='2':
        #     dataset = trainer.train_dataloader.dataset
        # else:
        #     dataset = trainer.train_dataloader.dataset.datasets
        dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
                # if args.data_type == 'wds_img':
                #     raw_dict = pl_module.state_dict()
                #     for k in raw_dict:
                #         if k.startswith('encoder.') or k.startswith('decoder.'):
                #             to_save_dict[k] = raw_dict[k]
                # else:
                #     to_save_dict = pl_module.state_dict()
                to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args, trainer,
                        to_save_dict,
                        f"{args.ckpt_dir}/{args.epoch_begin + trainer.current_epoch}_rwkv.pth",
                    )
                except Exception as e:
                    print('Error during my_save\n\n', e, '\n\n')

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n")
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            # if (args.epoch_begin + trainer.current_epoch) >= args.my_exit:
            #     exit(0)


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)
# -------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------

def sequences_collator(texts, w2i, max_seq_len):
    input_ids = []
    labels = []
    for t in texts:
        i, l = json.loads(t['text'])
        
        input_id = [w2i[w] for w in i]
        input_id += [w2i['<pad>']] * (max_seq_len - len(input_id))

        label = [w2i[w] if not w == '-1' else -1 for w in l]
        label += [-1] * (max_seq_len - len(label))

        input_ids.append(input_id)
        labels.append(label)
        
    return {
        'input_id': torch.LongTensor(input_ids),
        'label': torch.LongTensor(labels),
    }

# -------------------------------------------------------------------------------------

# time_slot = {}
# time_ref = time.time_ns()

# def record_time(name):
#     if name not in time_slot:
#         time_slot[name] = 1e20
#     tt = (time.time_ns() - time_ref) / 1e9
#     if tt < time_slot[name]:
#         time_slot[name] = tt

# class TOKENIZER():
#     def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
#         if 'list' in str(type(WORD_NAME)):
#             self.charMode = False
#             if WORD_NAME[0] == WORD_NAME[1]:
#                 from transformers import PreTrainedTokenizerFast
#                 self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
#             else:
#                 from transformers import GPT2TokenizerFast
#                 self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
#             self.vocab_size = len(self.tokenizer)
#         else:
#             self.charMode = True
#             with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
#                 self.word_table = json.load(result_file)

#             self.vocab_size = len(self.word_table)

#             self.stoi = {v: int(k) for k, v in self.word_table.items()}
#             self.itos = {int(k): v for k, v in self.word_table.items()}

#             self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

#     def refine_context(self, context):
#         context = context.strip().split('\n')
#         for c in range(len(context)):
#             context[c] = context[c].strip().strip('\u3000').strip('\r')
#         context = list(filter(lambda c: c != '', context))
#         context = '\n' + ('\n'.join(context)).strip()
#         if context == '':
#             context = '\n'
#         return context

#     def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
#         # out[self.UNKNOWN_CHAR] = -float('Inf')
#         lastChar = int(x[-1])

#         probs = F.softmax(out, dim=-1)

#         if self.charMode:
#             if self.itos[lastChar] == '\n':
#                 top_p = top_p_newline
#             else:
#                 top_p = top_p_usual
#         else:
#             top_p = top_p_usual

#         if os.environ["RWKV_RUN_DEVICE"] == "cpu":
#             probs = probs.numpy()
#             sorted_probs = np.sort(probs)[::-1]
#             cumulative_probs = np.cumsum(sorted_probs)
#             cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#             probs[probs < cutoff] = 0
#             if temperature != 1.0:
#                 probs = probs.pow(1.0 / temperature)
#             probs = probs / np.sum(probs)
#             out = np.random.choice(a=len(probs), p=probs)
#             return out
#         else:
#             sorted_probs = torch.sort(probs, descending=True)[0]
#             cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
#             cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
#             probs[probs < cutoff] = 0
#             if temperature != 1.0:
#                 probs = probs.pow(1.0 / temperature)
#             out = torch.multinomial(probs, num_samples=1)[0]
#             return out

# def MaybeIsPrime(number):
#     if FermatPrimalityTest(number) and MillerRabinPrimalityTest(number):
#         return True
#     else:
#         return False


# def FermatPrimalityTest(number):
#     if number > 1:
#         for time in range(3):
#             randomNumber = random.randint(2, number) - 1
#             if pow(randomNumber, number - 1, number) != 1:
#                 return False
#         return True
#     else:
#         return False


# def MillerRabinPrimalityTest(number):
#     if number == 2:
#         return True
#     elif number == 1 or number % 2 == 0:
#         return False
#     oddPartOfNumber = number - 1
#     timesTwoDividNumber = 0
#     while oddPartOfNumber % 2 == 0:
#         oddPartOfNumber = oddPartOfNumber // 2
#         timesTwoDividNumber = timesTwoDividNumber + 1

#     for time in range(3):
#         while True:
#             randomNumber = random.randint(2, number) - 1
#             if randomNumber != 0 and randomNumber != 1:
#                 break

#         randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

#         if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
#             iterationNumber = 1

#             while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
#                 randomNumberWithPower = pow(randomNumberWithPower, 2, number)
#                 iterationNumber = iterationNumber + 1
#             if randomNumberWithPower != (number - 1):
#                 return False

#     return True
