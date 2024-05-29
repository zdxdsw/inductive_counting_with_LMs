## Python Environment

```
cd <path_to_this_repo> &&
cd scripts/rwkv-v5 &&
python3 -m venv venv &&
source venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 &&
pip install pytorch-lightning==1.9.5 deepspeed==0.7.0 wandb ninja
pip install -r rwkv-v5_requirements.txt
```

## Train Models
```
python run.py --cuda 0 --task <task_name>
```
`<task_name>` can be choosen from [`scripts/causal_transformer/config_taskspecific.py`](https://github.com/zdxdsw/think_more_like_Transformers/blob/master/scripts/causal_transformer/config_taskspecific.py), e.g. `counting_samesymbol_mod10bos`.

## Test Models
```
python tester.py --handle <timestamp>
```
E.g., timestamp = 0522_103640

## Troubleshooting
1. deepspeed `no module named 'torch._six'` &#x2794; pip uninstall, then install deepspeed
2. deepspeed `no torch module` error &#x2794; pip3 install pytorch-lightning==1.9.5 deepspeed==0.7.0 wandb ninja **--no-build-isolation**
3. `subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1`; `nvcc fatal`; `ninja: build stopped: subcommand failed.` &#x2794; These are probably caused by unsupported cuda versions. Check `nvcc -V`. rwkv requires cuda>=11. We used cuda 11.8
