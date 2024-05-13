import json, time, random, os, sys
import numpy as np
import torch
#from torch.nn import functional as F
from datasets import load_dataset, concatenate_datasets
from functools import partial
sys.path.append("../")
from causal_transformer.utils import trim_task

# --------------------------------- moved from trainer.py from the original repo -------------------------------------
import os, math, time, datetime, subprocess
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

        val_data = load_dataset(
                        "text", 
                        data_files={"validation": f"{args.eval_data_path}/{trim_task(args.task)}/val.txt"}
                        )['validation']
    
        self.max_seen_len = max([len([x for x in json.loads(l['text'])[0] if x != "<pad>"]) for l in val_data])
        print(f"max_seen_len for {args.task} = {args.max_seen_len}")

        collator = partial(sequences_collator, 
                            w2i={w:i for i,w in enumerate(args.vocab)}, 
                            max_seq_len=args.ctx_len,
                        )

        val_dataloader = DataLoader(val_data, shuffle=False, batch_size=64, collate_fn=collator, pin_memory=True, num_workers=1, persistent_workers=False)

        self.val_dataloaders = {"val": val_dataloader}
        for ood_test_file in args.test_files:
            test_data = load_dataset("text",  data_files={ood_test_file: f"{args.eval_data_path}/{trim_task(args.task)}/{ood_test_file}.txt"})
            self.val_dataloaders[ood_test_file] = DataLoader(test_data[ood_test_file], shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=collator)


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args
       
        # LR schedule (disable decay)
        w_step = args.warmup_steps
        lr = args.lr_init
        
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
            else:
                param_group["lr"] = lr

        trainer.my_lr = lr
        trainer.my_wd = wd_now

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
                

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        real_epoch = args.epoch_begin + trainer.current_epoch
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

        if trainer.is_global_zero and real_step % args.eval_every_steps == 0:  # inference
            pl_module.eval()
            with torch.no_grad():
                # Inference on validation set
                avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = self.inference(
                    pl_module,
                    self.val_dataloaders['val']
                )

                print(f"""Epoch {real_epoch} Step {real_step} 
                        | Val Loss: {avg_loss} 
                        | Val Acc: {avg_acc} 
                        | Val Counting Acc: {avg_counting_acc} 
                        | Val Last Acc: {avg_last_acc}
                        | Val Unseen Len Acc: {avg_unseen_len_acc}"""
                    )
                
                # Inference on testing sets
                for ood_test_file, test_dataloader in self.val_dataloaders.items():
                    if ood_test_file == 'val': continue
                    avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = self.inference(
                        pl_module,
                        test_dataloader,
                    )
                    print(f"""Epoch {real_epoch} Step {real_step} {ood_test_file}.txt
                        | Test Loss: {avg_loss}
                        | Test Acc: {avg_acc}
                        | Test Counting Acc: {avg_counting_acc}
                        | Test Last Acc: {avg_last_acc}
                        | Test Unseen Len Acc: {avg_unseen_len_acc}"""
                    )
                    
            pl_module.train()


    def inference(self, pl_module, dataloader):
        counting_correct, counting_demo, last_correct, last_demo, unseen_len_correct, unseen_len_demo, correct, demo = 0, 0, 0, 0, 0, 0, 0, 0
        losses = []
        testing_output = {}
        k = 0
        for batch_idx, batch in enumerate(dataloader):
            (loss, 
                logits, 
                _counting_correct, 
                _counting_demo, 
                _last_correct, 
                _last_demo, 
                _unseen_len_correct,
            _unseen_len_demo) = pl_module.predict_step(batch, batch_idx)

            losses.append(loss)

            counting_correct += _counting_correct
            counting_demo += _counting_demo
            last_correct += _last_correct
            last_demo += _last_demo
            unseen_len_correct += _unseen_len_correct
            unseen_len_demo += _unseen_len_demo
            correct += (_counting_correct + _last_correct)
            demo += (_counting_demo + _last_demo)

            for input_id, gth_id, pred_id in zip(batch['input_id'], batch['label'], logits.argmax(dim=-1)):
                input_seq = [self.args.vocab[i] for i in input_id if self.args.vocab[i]!='<pad>']
                gth_seq = [self.args.vocab[gth_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1]
                pred_seq = [self.args.vocab[pred_id[i]] for i in range(len(gth_id)) if gth_id[i]!=-1][:len(gth_seq)]
                testing_output[k] = {
                    "input": " ".join(input_seq),
                    "gth": " ".join(gth_seq),
                    "pred": " ".join(pred_seq),
                }
                k+=1
        
        avg_loss = round(np.mean(losses), 4)
        avg_acc = round(correct/demo, 4)
        avg_counting_acc = round(counting_correct/counting_demo, 4)
        avg_last_acc = round(last_correct/last_demo, 4)
        if unseen_len_demo == 0:
            avg_unseen_len_acc = -1
        else:
            avg_unseen_len_acc = round(unseen_len_correct/unseen_len_demo, 4)
        print(f"inference loop {k} examples") 

        return avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc 
    

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset.datasets
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size

        # # Inference on validation set
        # avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = inference(
        #     pl_module,
        #     self.val_dataloader,
        #     criterion,
        #     accelerator.device,
        #     max_seen_len=self.args.max_seen_len,
        #     vocab=self.args.vocab,
        # )
        # print(f"""Epoch {epoch} Step {global_step} 
        #     | Val Loss: {avg_loss} 
        #     | Val Acc: {avg_acc} 
        #     | Val Counting Acc: {avg_counting_acc} 
        #     | Val Last Acc: {avg_last_acc}
        #     | Val Unseen Len Acc: {avg_unseen_len_acc}"""
        # )

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        if (trainer.is_global_zero) or ('deepspeed_stage_3' in args.strategy):  # save pth
            if (args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0) or (trainer.current_epoch == args.epoch_count - 1):
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

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps
        real_epoch = args.epoch_begin + trainer.current_epoch
        if trainer.is_global_zero: # inference
            pl_module.eval()
            with torch.no_grad():
                # Inference on validation set
                avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = self.inference(
                    pl_module,
                    self.val_dataloaders['val']
                )

                print(f"""Epoch {real_epoch} Step {real_step} 
                        | Val Loss: {avg_loss} 
                        | Val Acc: {avg_acc} 
                        | Val Counting Acc: {avg_counting_acc} 
                        | Val Last Acc: {avg_last_acc}
                        | Val Unseen Len Acc: {avg_unseen_len_acc}"""
                    )
                
                # Inference on testing sets
                for ood_test_file, test_dataloader in self.val_dataloaders.items():
                    if ood_test_file == 'val': continue
                    avg_loss, avg_acc, avg_counting_acc, avg_last_acc, avg_unseen_len_acc = self.inference(
                        pl_module,
                        test_dataloader,
                    )
                    print(f"""Epoch {real_epoch} Step {real_step} {ood_test_file}.txt
                        | Test Loss: {avg_loss}
                        | Test Acc: {avg_acc}
                        | Test Counting Acc: {avg_counting_acc}
                        | Test Last Acc: {avg_last_acc}
                        | Test Unseen Len Acc: {avg_unseen_len_acc}"""
                    )
                    
            pl_module.train()
            

@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

# ------------------------------------------------------------------------------------------------------------------------



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

