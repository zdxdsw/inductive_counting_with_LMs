from tqdm import tqdm
import sys, time

bar = tqdm(range(5), file=sys.stderr)

for i in range(5):
    time.sleep(1)
    bar.update(1)