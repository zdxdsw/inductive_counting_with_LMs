import random, time, os, pytz, argparse, yaml, sys, json
from config import *
from config_taskspecific import *
from tqdm import trange
from datetime import datetime
timezone = pytz.timezone('America/New_York') 

if os.path.exists('/data/yingshac/'): 
    os.environ['HF_HOME'] = '/data/yingshac/hf_cache'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="")
parser.add_argument('--sleep', type=int)
parser.add_argument('--port', type=str, default="29500")
parser.add_argument('--cuda', type=str, default="")
parser.add_argument('--turnoff_accelerator', action='store_true')
args = parser.parse_args()
if args.cuda: args.cuda = f"CUDA_VISIBLE_DEVICES=\"{args.cuda}\""

"""------------- Preparing configs -------------"""
if len(args.task): config = eval(f"{args.task}_Config")()
else: config = Basic_Config()

SEEDS = config.seed
if isinstance(SEEDS, int): SEEDS = [SEEDS]
for seed in SEEDS:
  date = datetime.now(timezone).strftime("%m%d_%H%M%S")
  config.date = date
  config.seed = seed

  if config.output_dir is not None:
    os.makedirs(os.path.join(config.ckpt_dir, config.date, "ckpts"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, config.date), exist_ok=True)

    if config.load_from_dir is not None:
      resume_from_config = json.load(open(os.path.join(config.output_dir, config.load_from_dir, "config.json"), "r"))
      for k in resume_from_config:
        if k not in ["warmup_steps", "learning_rate", "num_epochs", "save_every_steps", "eval_every_steps", "logging_steps", "load_from_dir", "date", "data_path"]:
          setattr(config, k, resume_from_config[k])
      
      if "task" in resume_from_config:
        args.task = resume_from_config['task']
      else:
        args.task = resume_from_config['data_path'].split("/")[-1] # compatible with old train.txt paths
        ## auto infer task from the incomplete run

    # dump config
    config.task = args.task
    C = {k:config.__getattribute__(k) for k in dir(config) if not k.startswith("__")}
    with open(os.path.join(config.output_dir, config.date, "config.json"), "w") as f:
      json.dump(C, f, indent=2)

  """------------- Sleep if needed -------------"""
  if args.sleep is not None:
    for i in trange(args.sleep, desc="Sleeping"):
      time.sleep(60)

  if not args.turnoff_accelerator:
    os.system("{} accelerate launch --main_process_port {} --num_process 1 trainer.py --date {} --task {} | tee {}".format(
        args.cuda,
        args.port,
        config.date,
        args.task,
        os.path.join(config.output_dir, date, "terminal.txt"),
    ))
  else:
    os.system("python trainer.py --date {} --task {} | tee {}".format(
        config.date,
        args.task,
        os.path.join(config.output_dir, date, "terminal.txt"),
    ))