#!/bin/bash
# Re-runs the listed measurement steps into a fresh output directory.
# Run from the repository root. Needs an NVIDIA GPU (measured on an RTX 5090 Laptop, sm_120), CUDA 13.1 toolkit
# (cuBLAS / cuBLASLt / NVRTC), PyTorch with CUDA, numpy and pynvml.
# The race is long (hours: the harness waits for the GPU to cool before every round, see docs/engine.md).
# Every step appends to $ENGINE_OUT/*.jsonl; records carry the hash of the code version that produced them.
set -e
export ENGINE_OUT="results/engine_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir "$ENGINE_OUT"   # fails if it already exists: a rerun never mixes with earlier evidence

python code/ozaki_engine.py diag                      # INT8 / FP16 / FP8 throughput diagnosis
python code/ozaki_engine.py selftest                  # bit-exact CRT core vs Python big integers
python code/ozaki_engine.py instr                     # positive and negative controls of every detector
python code/ozaki_engine.py check 2048 G,X,S,R,W      # accuracy, per-entry error-bound audit, full-matrix cross-check, full certificate
python code/ozaki_engine.py config                    # configuration table (moduli count and scaling bits per accuracy contract)
python code/ozaki_engine.py race 2048,4096,8192 G,X,S,R,W   # the race (n = 8192 uses split brackets, see docs/engine.md)
python code/ozaki_engine.py report                    # table of the race (latest record per cell)
python code/ozaki_engine.py trace                     # host-share trace of the timed path
python code/ozaki_engine.py persist 4096 G,X,W,R,S    # fixed-operand streams
python code/ozaki_engine.py audit                     # try to disprove the vendor-relative win (3 blocks per family)
