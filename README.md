> **Research moved to [Latent Space Reasoning](https://github.com/dl1683/Latent-Space-Reasoning).**
> All future work from this project continues in LSR, alongside persistent latent
> memory, long-context computation, new model construction/training/inference,
> and shared high-dimensional and neural mathematics. This repository preserves
> historical research; its old run/resume instructions are historical, not a
> current launch queue. Start with [LSR's current state](https://github.com/dl1683/Latent-Space-Reasoning/blob/main/STATE.md)
> and [unified evidence](https://github.com/dl1683/Latent-Space-Reasoning/blob/main/docs/UNIFIED_EVIDENCE.md).
> Consolidation does not expand the scientific scope of the results below.

# Matrix Numbers: arithmetic with the matrix as the atom

**What if the matrix, not the number, were the atom of arithmetic?**

Every number system we use treats the scalar as the unit: the axis is built around 1, operations are built around units of 1, and matrices are arrays of scalars bolted on top. The hardware that matters most has already moved on. A tensor core's instruction is a matrix product; a TPU's native operation is a 128x128 multiply. Mathematics and notation are the part of the stack that still thinks in scalars.

This project asks, concretely and with measurements, what arithmetic looks like when the matrix is the primitive object and the scalar is just the 1x1 case: a representation plus native operations that make matrix computation fundamentally cheaper. We work on the mathematics first, build what the mathematics suggests, and measure it against the fastest accuracy-screened vendor variants tried on this setup.

> **Status: active research, work in progress.** The results below are real, measured, and independently re-checked against the raw records. Each states what was measured, on what hardware, and what is not yet established. Nothing here is a finished product.

## Results at a glance

| # | Result | Rigor label | Where |
|---|---|---|---|
| 1 | FP64 matrix multiplication built from INT8 tensor-core products (residue / Chinese Remainder representation): about 11-17x faster than native FP64 and about 1.5-3x faster than the vendor's own dynamic emulation on a laptop GPU, n >= 4096 | EMPIRICAL (one machine, cool-start, sampled accuracy) | [docs/engine.md](docs/engine.md) |
| 2 | Exact counts of low-precision products needed for one higher-precision product: 11 published counts certified optimal in their model, an exact 32-vs-36 improvement for pure-FP8 hardware, and a certified 16-product optimum for exact FP64 significands with one INT8 product per modulus | Theorems with machine-checked exhaustive searches; independent proof re-check in progress | [docs/product_count_theory.md](docs/product_count_theory.md) |
| 3 | A verified map of which structured-matrix classes (low displacement rank, butterfly, Monarch, hierarchical) are closed under multiplication, inversion and Kronecker products, with exact dimension counts | KNOWN + EMPIRICAL (exact finite checks); no new theorem claimed | [docs/structured_matrix_map.md](docs/structured_matrix_map.md) |

## Why this might matter

Computing eras are defined by their primitives. We suspect that the primitives we inherited (scalars, entry-wise arithmetic, dense arrays) may be quietly limiting how efficiently high-dimensional computation can be done, and that better primitives exist. This is a thesis we are testing, not a result we have shown; see [docs/vision.md](docs/vision.md). Matrix products sit under scientific simulation, signal processing, optimization and machine learning alike, so even modest structural gains compound. The work is general-purpose mathematics; its implications for high-dimensional computing, machine learning included, are implications and not the definition of the project.

## 1. A matrix-native FP64 engine on integer tensor cores *(early, promising)*

Double-precision matrix multiplication on GPUs whose double-precision units are deliberately slow, built from small-integer matrix products and the Chinese Remainder Theorem: each scaled matrix is held as a stack of residue matrices, multiplied by exact INT8 tensor-core products, and reconstructed once. The number of residue planes is chosen from the data by an a-priori error rule.

Measured on one laptop GPU (RTX 5090 Laptop, native FP64 about 0.69 TFLOPS), square FP64 inputs, n >= 4096, cool-start (the harness waits for the GPU to cool before every round), paired against native FP64 in every round, drift-checked, and independently re-audited from the raw records:

| Comparison (n >= 4096) | Speedup of our engine |
|---|---|
| vs native FP64 (cuBLAS DGEMM) | about 11-17x (11.2x to 17.0x across the 8 valid cells) |
| vs the vendor's own emulation, dynamic rule, fastest variant tried (cuBLAS 13.1) | about 1.5-3x (race medians; independent audit means 1.7-2.8x) |
| vs the fastest calibrated vendor variant (fixed mantissa or dynamic offset; needs a calibration the engine does not) | about 1.2-1.7x (1.21x to 1.23x at n = 4096 for families R and G) |

The engine uses 16-20 INT8 products where the vendor's accuracy-matched path uses roughly 36-78. The advantage is structural (fewer products), not a faster kernel. The accuracy contract is statistical, not a guarantee, as explained next.

**What is not yet established.**
- Accuracy is verified on sampled entries; the error bound (Lemma L1) holds at every entry, but the rule that picks the number of residue planes checks it on a probe set. A full-matrix certificate exists and was run at n = 2048 but is not in the timed path.
- Results are cool-start, boost-clock numbers, not sustained-load numbers. Behaviour under sustained thermal throttling was not measured.
- The margin over the calibrated vendor variant is small (about 1.2x for two families). The independent audit blocks did not satisfy their own pairing check, so its confidence bounds are indicative only.
- At n = 2048 the engine is faster in median but the intervals overlap with the best vendor variant (host launch jitter of a roughly 3 ms call); no win is claimed there. At n = 8192, two of five families (S, W) could not be validated under the drift rule, and n = 8192 uses split timing brackets (an amended protocol).
- One GPU, one driver stack. n = 16384 was not run.

Raw records and a one-command replay are in [results/engine/](results/engine/).

## 2. Exact theory of low-precision product counts *(machine-checked cases, independent proof re-check in progress)*

How many low-precision matrix products are needed to build one higher-precision product? We prove a lower-bound theory (a value-set Sylvester bound, a structure theorem for the tight case, ratio lemmas, per-modulus bounds), then decide small cases exhaustively.

- 11 of the published product counts (Hayashi et al., FP4-based emulation) are certified optimal in their model; a 12th, the INT8 x INT8 count of 6, is in its final verification and is not claimed yet.
- For pure-FP8 hardware, a mixed-cost modulus selection gives 32 FP8 products instead of the published 36 for FP64-level products; every per-modulus identity and three end-to-end shapes were verified exactly. No wall-clock claim is made, and where INT8 hardware exists INT8 needs 15-16.
- Exact 54-bit significand products need 16 INT8 products when each product serves one modulus; the general bilinear floor is 13, and the gap 13 vs 16 is open.
- A cost law for chains of products held in residue form, and a calculator that maps accuracy contracts to residue-plane counts and throughput ceilings ([results/calculator/](results/calculator/)).

## 3. A controlled map of structured-matrix classes *(Floor tier)*

Which classes of structured matrices (low displacement rank, butterfly, Monarch, hierarchical) are closed under multiplication, inversion and Kronecker products with bounded descriptor growth, and which provably are not, with exact dimension counts and scripted checks. **No new theorem is claimed** beyond the cited literature; the value is a verified map of what is known, where each statement is exact, conditional or open, and which cell of the map (expressive and closed under everything with bounded growth) remains empty.

## What we are building toward
- A certified accuracy contract: a cheap all-entry error bound that selects the minimal residue plan for each input.
- Staying in matrix-native form across chains of operations, where measurements justify it. Our own cost law says residue chains do not save products; they save conversion bytes, so the engine must be judged on time, bytes and error together.
- Benchmarks that earn respect: an open, accuracy-matched shoot-out against every vendor emulation mode, and one public scientific kernel (density-fitted exchange in quantum chemistry is the leading candidate). Official TOP500 and Green500 lists exclude integer-emulated arithmetic, so we publish an open reproducible corpus instead.
- A mathematical account of when matrix-native representations beat entry-wise ones. See [docs/status_and_roadmap.md](docs/status_and_roadmap.md).

## How we verify
- **Raw records.** Every timing cell is stored in full (`results/engine/race_results.jsonl`: all valid and invalid rounds, clocks, temperatures, error measurements, the code hash). Invalid rounds are kept, not dropped. Cells that fail the drift rule are reported as invalid and excluded from conclusions.
- **Replay.** `results/engine/replay.sh` re-runs the listed measurement steps into a fresh output directory; `python code/ozaki_engine.py report` reprints the race table from the raw records (latest record per cell).
- **Evidence status.** Headline engine numbers were recomputed independently from the raw records. The original measurement code could not be preserved bit-for-bit (see Reproducibility). Theorem statements and finite machine checks are provided; an independent proof re-check is pending. No new theorem is claimed for the structured-matrix map.
- **Rigor labels.** Every claim carries PROVED, EMPIRICAL, CONJECTURE, SPECULATION (or KNOWN for cited literature), and nothing is promoted past its label. In this repository the finite statements are machine-checked, but a claim is not labelled PROVED until an independent proof re-check has passed; statements still awaiting it say so.
- **Baselines are the fastest accuracy-screened vendor variants tried on this setup.** Native FP64, the vendor's emulation at its fastest tried configuration meeting the same accuracy contract, and a calibrated fixed-mantissa mode.
- **Selftests.** The code carries bit-exact integer tests of the residue core and positive and negative controls for every detector that supports a claim.

## Reproducing
```
python code/ozaki_engine.py report          # race table from the raw records (CPU only; imports torch)
bash results/engine/replay.sh               # re-run the measurements (needs an NVIDIA GPU, CUDA 13.1)
python code/contract_plane_table.py --selftest
python code/floor_ldr_checks.py r3          # exact structured-matrix checks, about a second
python code/digit_gemm.py awe               # reproduces a published INT8 x INT8 six-product scheme exactly
```
Dependencies: Python 3, numpy; the engine also needs PyTorch with CUDA, pynvml and the CUDA 13.1 toolkit.

## Honesty policy
Negative and inconclusive results are kept (the 2048 cells, the 8192 cells excluded under the thermal drift rule and queued for cooler re-runs, the long-stream fixed-operand test, the FP4 modulus selection that did not beat the published count). Timings state hardware, thermal regime, baselines and accuracy contract. Where our result is an engineering gain from using fewer products rather than a new arithmetic, we say so.

## License
MIT, see [LICENSE](LICENSE).

