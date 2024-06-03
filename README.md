
## &#x1f31f; This is the code repo for experiments performed in [Language Models Need Inductive Biases to Count Inductively](https://arxiv.org/abs/2405.20131) &#x1f31f;

## File Structure
In `/scripts`, we maintain separate folders for different architecture types. Note, LSTM and RNN are subsumed in `/scripts/s4`.

## Python Environments

To support reproducibility for individual sets of experiments, `mamba` and `rwkv` have their own environments, while `causal_transformer` and `s4` use a shared env. Thus, we provide instructions for building three environments.

Here's how you setup the shared environment for `causal_transformer` and `s4`. 
```
cd <path_to_this_repo> &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install -r requirements.txt &&
cd scripts/s4 &&
pip install -r s4_requirements.txt
```
Please click these links for building [`mamba`](https://github.com/zdxdsw/think_more_like_Transformers/blob/master/scripts/mamba/README.md) and [`rwkv`](https://github.com/zdxdsw/think_more_like_Transformers/blob/master/scripts/rwkv-v5/README.md) environments.


## Generate Data
For examples of the input-output formats, [there](https://github.com/zdxdsw/inductive_counting_with_LMs/tree/master/data/rasp_primitives) are validation and OOD testing files for each task.

Our training data is generated in this [notebook](https://github.com/zdxdsw/inductive_counting_with_LMs/blob/master/notebooks/counting.ipynb).

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

Remember to specify `output_dir, ckpt_dir, hf_cache_dir` in `config.py`.

**Training command:**
```
cd scripts/causal_transformer && # or cd scripts/s4
python run.py --task <task_name> --cuda 0 --port 29500
```

**Notes**

- `<task_name>` can be choosen from [`scripts/causal_transformer/config_taskspecific.py`](https://github.com/zdxdsw/think_more_like_Transformers/blob/master/scripts/causal_transformer/config_taskspecific.py), e.g. `counting_samesymbol_mod10bos`.

- Model ckpts will be saved to `ckpt_dir` specified in `config.py`
Model outputs during validation will be saved to `output_dir`. Specifically, each run will create its own folder under `output_dir` named by the timestamp, which can be passed to `tester.py` through the argument "handle".

- If you're running multiple jobs on the same machine, use different ports. Otherwise, accelerator will complain about busy port.

## Test Models

```
python tester.py --handle <timestamp>
```
E.g., timestamp = 0522_103640


## Cite Us &#x1f64f;
```
@article{chang2024language,
  title={Language Models Need Inductive Biases to Count Inductively},
  author={Chang, Yingshan and Bisk, Yonatan},
  journal={arXiv preprint arXiv:2405.20131},
  year={2024}
}
```

## Acknowledgements
- Implementation of causal Transformer, as well as its positional embedding variants, is borrowed heavily from huggingface's implementation of [gpt-2](https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/gpt2/modeling_gpt2.py), [t5](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) and [llama](https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/llama/modeling_llama.py).
- We give credit to the [official S4 repo](https://github.com/state-spaces/s4/blob/main/models/s4/s4.py) for implementation of s4.
- We give credit to the [official rwkv repo](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v5) for implementation of rwkv.
- We give credit to the [official mamba repo](https://github.com/state-spaces/mamba) for implementation of mamba, as well as the [mamba-chat repo](https://github.com/redotvideo/mamba-chat) for setting up the mamba environment. 
