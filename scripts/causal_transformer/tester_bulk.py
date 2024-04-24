import os, json
from tqdm import tqdm
os.environ['HF_HOME'] = '/data/yingshac/hf_cache'


## 
output_dir = "output"
commands = []
for handle in sorted(os.listdir(output_dir)):
    #if handle >= "0421_202807" or handle <= "0421_111336": continue
    config = json.load(open(f"{output_dir}/{handle}/config.json", "r"))
    if 'task' in config:
        task = config['task']
    elif 'data_path' in config:
        task = config['data_path'].split("/")[-1]
    else:
        print("Cannot find task from the config file: ", handle)
        continue
    if 'rotary_posemb' in config and config['rotary_posemb']: continue
    if task in ["counting_samesymbol_shiftedstart"]:
        loop10 = False
        for k in ['rotary_posemb_shift', 'rotary_posemb_rdmz', 'absolute_posemb_shift', 'absolute_posemb_rdmz']:
            if k in config and config[k]: loop10 = True
        loop = 10 if loop10 else 1
        test_files = ["ood_test"]
        if "test_files" in config: test_files = config["test_files"]
        command = "python tester.py --handle {} --loop {} --test_files \"{}\"".format(handle, loop, " ".join(["val"]+test_files))
        print(command, task)
        commands.append(command)

proceed = input('Proceed? (y/n): ').lower().strip() == 'y'
if proceed:
    for command in tqdm(commands):
        print(command)
        os.system(command)
else:
    print("Aborted.")