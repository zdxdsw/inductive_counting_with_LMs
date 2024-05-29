
This is the code repo for experiments performed in *Language Models Need Inductive Bias to Count Inductively*.

## File Structure
In `/scripts`, we maintain separate folders for different architecture types. Note, LSTM and RNN are subsumed in `/scripts/s4`.

## Python Environments

To support reproducibility for different sets of experiments, separate environments are used for `mamba` and `rwkv`. `causal_transformer` and `s4` share the same env. Thus, we provide instructions for building three environments.

Here's how you setup the environment for `causal_transformer` and `s4`. 
```
cd <path_to_this_repo> &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install -r requirements.txt &&
cd scripts/s4 &&
pip install -r s4_requirements.txt
```
Please click these links for building `mamba` and `rwkv` environments.

## Train Models
If this is the first time you use accelerate, and you haven't configured it, please do:
`accelerate config`, and config accordingly.
```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Training command:
```
cd scripts/<architecture_type>
python run.py --task <task_name> --cuda 0 --port 29500
```
Model ckpts will be saved to `ckpt_dir` specified in `config.py`
Model outputs during validation will be saved to `output_dir`. Specifically, each run will create its own folder under `output_dir` named by the timestamp, which can be passed to `tester.py` through the argument "handle".

## Test Models

Testing command:
```
python tester.py --handle 0522_103640
```

## Cite Us &#x1f64f;
