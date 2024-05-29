## Python Environment
```
cd <path_to_this_repo> &&
cd scripts/mamba &&
python3 -m venv venv &&
source venv/bin/activate &&
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 &&
pip3 install --upgrade pip &&
pip3 install wheel &&
pip3 install packaging &&
pip3 install transformers==4.35.0 &&
pip3 install mamba-ssm==1.0.1 &&
pip3 install accelerate==0.25.0 &&
pip3 install bitsandbytes==0.41.3 && 
pip3 install scipy==1.11.4 &&
pip3 install causal-conv1d==1.0.0
```

## Troubleshooting
1. `ImportError: ....../mamba/venv/lib64/python3.9/site-packages/selective_scan_cuda.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops10zeros_like4callERKNS_6TensorEN3c108optionalINS5_10ScalarTypeEEENS6_INS5_6LayoutEEENS6_INS5_6DeviceEEENS6_IbEENS6_INS5_12MemoryFormatEEE`
This may happen at `import causal_conv1d_cuda`; Or `ImportError: ....../mamba/venv/lib64/python3.9/site-packages/selective_scan_cuda.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv` This may happen at `import selective_scan_cuda` &#x2794; Check your cuda version. mamba requires cuda >= 11. We used cuda 11.8
