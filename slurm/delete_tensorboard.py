import os
from tqdm import tqdm

# delete tensorboard
output_dir = "/home/yingshac/workspace/llms_do_math/scripts/causal_transformer/output"
commands = []
for h in tqdm(os.listdir(output_dir)):
    if "tensorboard" in os.listdir(f"{output_dir}/{h}"):
        command = f"rm -r {output_dir}/{h}/tensorboard"
        commands.append(command)
        

output_dir = "/home/yingshac/workspace/llms_do_math/scripts/s4/output"
for h in tqdm(os.listdir(output_dir)):
    if "tensorboard" in os.listdir(f"{output_dir}/{h}"):
        command = f"rm -r {output_dir}/{h}/tensorboard"
        commands.append(command)

print("\n".join(commands))

proceed = input('Proceed? (y/n): ').lower().strip() == 'y'
if proceed:
    for command in tqdm(commands):
        os.system(command)
else:
    print("Aborted.")