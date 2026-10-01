# The FP64-on-INT8 residue engine

**Claim in one line.** On a GPU whose double-precision units are deliberately slow, an FP64 matrix product can be computed from 16-20 exact INT8 tensor-core products held in a residue (Chinese Remainder) representation, about 11-17x faster than native FP64 and about 1.5-3x faster than the vendor's own dynamic emulation, at n >= 4096 on one laptop GPU, with a statistical (sampled) accuracy contract. Label: **EMPIRICAL**. Code: [`code/ozaki_engine.py`](../code/ozaki_engine.py). Data: [`results/engine/`](../results/engine/).

This is an engineering result about a representation, not a new exponent: it uses about 2.5x fewer INT8 products than the vendor's path, it needs no new arithmetic, and the margin against a calibrated vendor mode is small. It matters here because it is the first place in this project where changing the representation of a matrix product changed a measured cost against the fastest accuracy-screened vendor variants tried on this setup.

## 1. Prior art and what is different

- Ozaki, Uchino, Imamura, "Ozaki Scheme II" (arXiv 2504.08009; GEMMul8): CRT with pairwise-coprime moduli, about 14-16 INT8 products for FP64-level accuracy. It does not use inside (contraction-index) scaling, does not give an a-priori componentwise error contract that picks the number of moduli, and was not compared against cuBLAS 13.1's built-in emulation on this class of part.
- Abdelfattah, Dongarra, Fasi, Mikaitis, Tisseur (arXiv 2506.11277): rounding-error analysis of Ozaki-I slicing; badly scaled inputs defeat 18 slices. Does not treat the CRT variant or a cost-aware choice of the number of slices.
- Demmel, Henry, Kozachenko, Langou, Li, Riedy, Vanover, "How to grade the accuracy of the BLAS" (arXiv 2609.12307, section 5.2.1): both Ozaki variants pass the graded-exponent test only with inner scaling; a rotated test defeats inner scaling. No speed measurements.
- NVIDIA cuBLAS 13.x fixed-point FP64 emulation (the baseline); Ootomo, Ozaki, Yokota (arXiv 2306.11975).

What this engine adds: inside scaling, a data-driven number of moduli chosen by an a-priori error rule, a cost-honest accuracy-matched comparison against the strongest vendor configuration, and a theory-facing configuration table.

## 2. Method

Inputs are finite float64 matrices A (m x k) and B (k x n).

1. **Statistics.** Per-contraction-index exponents give an inside scaling gamma_l = floor((b_l - a_l)/2); row and column exponents r_i, c_j follow. Non-finite input falls back to native FP64.
2. **Scaled integers.** A'_il = round(A_il 2^(p - r_i + gamma_l)) and B'_lj = round(B_lj 2^(p - c_j - gamma_l)); entries are at most 2^p in magnitude with quantisation error at most 1/2.
3. **Residues.** A' and B' are reduced exactly (integer arithmetic on the GPU, from the FP64 bits) modulo each of N pairwise-coprime moduli (256, 255, 253, 251, 247, ... up to 32 moduli), stored as signed int8 with the contraction index contiguous.
4. **N INT8 GEMMs** with INT32 accumulation (cuBLASLt, TN layout), each reduced mod its modulus. Exact for k <= 131071.
5. **Reconstruction.** Mixed-radix (Garner) reconstruction of A'B' in multi-limb integers, one correctly rounded conversion to FP64, and rescaling by powers of two.

**Lemma L1 (elementary; independent re-check pending).** For finite inputs whose results lie in the normal binary64 range (an absolute underflow allowance of 2^-1075 applies below it), if the CRT stage is exact (the product magnitude is below half the modulus product), then at every entry

`|C_hat_ij - C_ij| <= E_ij(p) + 2^-53 |C_hat_ij|`, with `E_ij(p) = (1/2) 2^(r_i - p) s^B_j + (1/2) 2^(c_j - p) s^A_i + (1/4) k 2^(r_i + c_j - 2p)`,

where s^A_i = sum_l |A_il| 2^gamma_l and s^B_j likewise. A range bound gives the largest safe p for each N.

**A-priori rule.** N* is the smallest N in [6, 32] such that E_ij(p_max(N)) <= 2^-52 D_ij / 2 on a probe set of about 9,000 of the n^2 entries, with D_ij = sum_l |A_il||B_lj|. If N* would exceed 32 the call falls back to native FP64. N*, p and all scalings use only the inputs.

**Rigor of the contract.** The bound L1 holds at every entry whose result lies in the normal binary64 range, with an absolute underflow allowance of 2^-1075 below it. The comparison with the target tolerance is evaluated on the probe set, so the number of moduli is chosen by a **sampled (statistical) contract, not a guarantee**. Two rigorous companions exist: (i) the data-independent configuration table (section 6), a worst-case contract relative to the row and column scales for finite inputs whose results lie in the normal binary64 range (an absolute underflow allowance of 2^-1075 applies below it); (ii) an optional full-matrix certificate (one FP16 tensor-core product of scaled magnitudes gives a lower bound on every D_ij), which at n = 2048 confirmed the sampled number of moduli in 5 of 5 families with 0 uncertified entries, at an unoptimised cost of 7-12 ms (one family includes first-call compilation time) against about 3 ms for the engine. (The 528 ms record for family G at n = 2048 includes first-call compilation and is excluded from this range.) It is not in the timed path, and it assumes a stated bound on tensor-core FP32 accumulation error.

## 3. Setup of the measurements

- **Machine.** RTX 5090 Laptop GPU (sm_120, 82 SMs), native FP64 about 0.69 TFLOPS peak; PyTorch 2.12 (dev, CUDA 12.8 build) plus the system CUDA 13.1 toolkit for cuBLAS 13.1 and cuBLASLt (via ctypes) and NVRTC for the engine's own kernels.
- **Baselines.** (a) Native FP64 (`torch.matmul`, cuBLAS DGEMM) in every round, before and after. (b) cuBLAS 13.1 DGEMM in FP64 fixed-point emulation mode, screened over 27 variants per cell (strategy, workspace {library default, 2 GiB, 8 GiB}, mantissa cap, special-values flag, and as calibrated oracle-class variants fixed mantissa b in {40, 48, 56, 64} and dynamic offsets). Dynamic-class variants use the library's own a-priori rule; fixed-mantissa variants need a calibration against the native error. The carried set per cell is the out-of-the-box call plus the 3 fastest other dynamic variants plus the 2 fastest calibrated variants. "Best of 27" would overstate this: only the fastest variants after accuracy screening were timed in the main rounds.
- **Input families** (float64, seeded, n x n): G Gaussian; X graded exponents; S contraction-index similarity (A = G D, B = D^-1 G'); R row and column scalings 2^s, 2^t, s,t in [-20, 20]; W ill-scaled and rotated (after Abdelfattah et al.), which inside scaling cannot balance.
- **Accuracy matching.** On a shared 49 x 49 sample against a correctly rounded reference (Dekker TwoProduct plus exact summation): E1 = max|C_hat - C_ref| / max|C_ref| and E2 = max componentwise error relative to (|A||B|)_ij. A method matches in a cell if its E1 and E2 are no larger than native's.
- **Timing and drift rule.** Each round is [native] [all other methods in random order] [native]; a round is valid only if the two native timings agree within 5%. Seven valid rounds are required per cell, no outlier removal. Every round starts only after the GPU is reported cool (at most 74 C, waiting up to 150 s), so these are **cool-start, boost-region** numbers. A 3 ms call under heavy ambient CPU load has launch jitter; this matters at n = 2048.
- **Cost accounting.** Everything is inside the timed region: statistics, the rule, residue conversion, N INT8 GEMMs, reductions, reconstruction, and the host read of (N, p). Buffers are preallocated per size (3 N n^2 + 4 n^2 bytes for the engine at capacity 20 moduli, against the vendor's 2 or 8 GiB user workspace). Nothing is amortised across calls.

## 4. Results

Speedups over native FP64 are medians of per-round ratios; "own" = the engine with the a-priori configuration. N = number of moduli (p = scaling bits). "Oracle N" is the smallest N that matches native accuracy on the sample (needs a calibration, so it is secondary).

| n | family | N (p) / oracle N | vendor out of the box | best vendor dynamic | best calibrated vendor variant (fixed mantissa or dynamic offset) | **own** | own, oracle N | time ratio t_own/t_vendor (lower is better) | non-overlapping IQRs and > 10% faster? |
|---|---|---|---|---|---|---|---|---|---|
| 2048 | G | 16 (59) / 14 | 3.17 | 6.46 | 8.15 | 7.02 | 9.63 | 0.92 [0.91, 1.21] (1.09x speedup) | no (overlap) |
| 2048 | X | 18 (68) / 16 | 2.55 | 4.63 | 6.49 | 6.54 | 8.77 | 0.70 [0.70, 0.99] (1.43x speedup) | no |
| 2048 | S | 16 (59) / 14 | 0.82 | 3.43 | 3.95 | 6.78 | 9.37 | 0.51 [0.49, 0.96] (1.96x speedup) | no |
| 2048 | R | 16 (58) / 15 | 2.16 | 6.51 | 8.28 | 6.43 | 8.90 | 1.01 [0.97, 1.35] (0.99x speedup) | no |
| 2048 | W | 20 (74) / 18 | 0.86 | 3.99 | 5.64 | 6.21 | 7.91 | 0.64 [0.64, 0.97] (1.56x speedup) | no |
| 4096 | G | 16 (58) / 14 | 6.07 | 8.51 | 10.76 | **13.21** | 16.30 | 0.66 [0.60, 0.66] (1.52x speedup) | **yes** |
| 4096 | X | 18 (67) / 15 | 0.95 | 5.05 | 8.67 | **12.58** | 15.80 | 0.40 [0.39, 0.41] (2.50x speedup) | **yes** |
| 4096 | S | 16 (58) / 14 | 0.95 | 4.36 | 5.01 | **13.47** | 16.70 | 0.32 [0.24, 0.33] (3.12x speedup) | **yes** |
| 4096 | R | 16 (58) / 15 | 5.62 | 8.61 | 11.16 | **13.52** | 15.56 | 0.64 [0.60, 0.69] (1.56x speedup) | **yes** |
| 4096 | W | 20 (74) / 18 | 0.95 | 5.10 | 7.17 | **11.24** | 13.53 | 0.45 [0.43, 0.46] (2.22x speedup) | **yes** |
| 8192 | G | 16 (58) / 14 | 6.23 | 8.77 | 10.45 | **17.04** | 20.96 | 0.51 [0.39, 0.55] (1.96x speedup) | yes (split brackets) |
| 8192 | X | 18 / 15 | n/a | 4.70 | 8.87 (split brackets) | **13.92** | 18.89 | 0.33 [0.30, 0.36] (3.03x speedup) | yes (split brackets) |
| 8192 | R | 16 / 15 | n/a | 8.60 | 11.22 (split brackets) | **16.67** | 18.64 | 0.52 [0.49, 0.53] (unpaired) (1.92x speedup) | yes (split brackets) |
| 8192 | S, W | | | | | | | invalid under the drift rule | excluded |

Reading the table honestly:
- The cool-start native baseline ran at 1.01-1.05 times the nominal 0.688 TFLOPS time everywhere, i.e. the cool regime.
- 15 cells were attempted; 13 are valid and 2 invalid (using the latest record per cell). At n = 8192 the originally specified single native bracket failed (self-heating makes native drift about 9%), so n = 8192 uses split brackets, an **amended protocol**; the 8192 S and W cells never reached seven valid rounds.
- At n = 2048 the engine is faster than the best dynamic vendor variant in median (except R), but the intervals overlap, so no win is claimed.
- The vendor's out-of-the-box call runs at or below native speed on S and W (0.82-0.95x), and on X at n = 4096 (0.95x); we read this as its fallback on wide exponent ranges. Raised-cap and workspace variants do not fall back, so that observation concerns only the out-of-the-box call.
- Against the fastest calibrated vendor variant (fixed mantissa b = 48 for G and R, fixed b = 64 or a dynamic offset for X and W; these need a calibration against the native error), the engine is faster by about 1.2-1.7x on G, X, R, W (1.21x R and 1.23x G at n = 4096, 1.45x X, 1.57x W; 1.63x G at 8192) and by a larger factor on S. The independent audit measured 1.19x on G (nominal lower bound 1.07).
- Both sides over-deliver on accuracy in the a-priori mode (own about 20-1000x more accurate than native on the sample; the vendor's dynamic rule about 13x). The calibrated "oracle" columns compare minimal configurations and are the accuracy-matched comparison proper.

### The advantage is structural

The vendor's retained-mantissa recommendation is 62 bits (G, R), 79-81 (X, W) and 91 (S), which is Ozaki-I slicing with about 8-12 slices, i.e. 36-78 INT8 products, against 16-20 residue products here. The gain comes from the product count.

### INT8 throughput diagnosis

An earlier version of this work passed a row-major right operand to `torch._int_mm`, which is 2.4-3.8x slower than the same kernel with a column-major operand; INT8 was therefore measured at the same rate as FP16. Measured in the loaded state (T-ops/s, median of 11 interleaved rounds):

| n | FP16 (fp32 acc) | FP8 e4m3 | `_int_mm` row-major B | `_int_mm` column-major B | cuBLASLt TN INT8 | GemmEx INT8 | SM clock |
|---|---|---|---|---|---|---|---|
| 2048 | 27.5 | 42.9 | 46.6 | 116.0 | 96.0 | 102.7 | 1402 MHz |
| 4096 | 42.4 | 79.8 | 43.4 | 140.0 | 140.6 | 140.0 | 1087 MHz |
| 8192 | 29.4 | 51.7 | 29.1 | 110.2 | 110.0 | 110.9 | 697 MHz |

A bare `mma.sync` INT8 loop reaches 401.4 TOPS at an inferred 1816 MHz (2696 ops/clock/SM); FP16 and e4m3 reach 100.5 and 201 TFLOPS (ratios 4:1:2). In sustained GEMMs the limiter is the SM clock under thermal slowdown (697-1402 MHz at 85-88 C). INT8 peak per clock is 512x the FP64 peak per clock on this part. Raw data: `results/engine/int8_diag.json`.

### Correctness checks

- 48 integer-exactness cases (N = 8, 14, 20, 30; two scalings; two shapes; Gaussian, graded and wide data including zeros, subnormals and 2^+-90 exponent ranges) reproduce the Python big-integer product bit for bit. Device statistics equal a numpy reference; graph replay equals eager execution (`selftest.json`).
- The a-priori configuration matched native accuracy in all five families at n = 1024 and n = 2048 (`check_*.json`). Lemma L1 audit: 0 violations in every family (median bound-over-error 38 for G, 4.5 for X, 38 for S, 53 for R, 15 for W). Inside scaling exactly undoes the similarity of family S (bit-identical N, p and errors to G); it saves 7 moduli on S (16 with it, 23 without).
- Full-matrix checks at n = 4096 (audit): componentwise E2 of the a-priori result 2.2e-17 (G), 3.0e-17 (W), 1.7e-16 (X) against native 6.8e-16, 1.1e-14, 1.2e-14. The sample-matched *oracle* configurations are not always matched over the whole matrix (X: own oracle 1.7e-14 and a vendor offset mode 7.0e-14 exceed native's 1.2e-14). The reference for full-matrix numbers is the engine's own N = 32 result (quantisation error about 2^-100, CRT stage bit-exact by the selftest), not an independent library.
- Data-independent configuration test: 6 of 6 configurations (sigma 2^-40 and 2^-50 on G, X, W) meet their guaranteed contract with 0 violations (`config_test.json`).

## 5. Cost breakdown and the host-bottleneck test

Warm eager profile at n = 4096, N = 16: statistics 2.25 ms (15%), residues 1.62 ms (11%), INT8 GEMMs plus reductions 10.2 ms (68%; 216 TOPS in the stage, about 54% of the bare loop's 401.4 TOPS), reconstruction 0.88 ms (6%); 14.9 ms total. At n = 8192: 13.1 / 10.1 / 108.8 / 5.8 ms, GEMM stage 79%. Analytic DRAM traffic is about (72 + 12 N) n^2 bytes per call (4.4 GB at n = 4096, N = 16).

A test specified in advance asked whether host-side reading and dispatch was a bottleneck (if it took 20% or more of wall time, a device-resident dispatch would be next; below 10% the premise is rejected). Measured share: 1.5% (G) and 1.2% (W). The premise is rejected. End to end at n = 4096: 14.3 ms (G) and 17.6 ms (W) against 22.1 ms and 40.0 ms for the best dynamic vendor variant (`trace_results.jsonl`).

## 6. Configuration table (interface to the theory)

`results/engine/config_table.json` (schema `ozaki_crt_int8_config_v1`) lists, for inner dimension k from 1024 to 16384 and N = 6..32, the largest data-independent p and the guaranteed scaled contract `|C_hat - C|_ij <= sigma k 2^(r_i + c_j) + 2^-53 |C_hat_ij|` for finite inputs whose results lie in the normal binary64 range (an absolute underflow allowance of 2^-1075 applies below it), with sigma = 2^-p (1 + 2^(-p-2)). Example (k = 8192): N = 15 gives p = 53, N = 16 gives p = 58, N = 12 gives p = 41. This contract is relative to row and column scales, not to the componentwise |A||B|; the theory documents ([product counts](product_count_theory.md)) use it to choose (N, p).

## 7. Audit that tried to disprove the win

At n = 4096 for G, W, X: native, best vendor dynamic, best vendor fixed or offset variant, own a-priori, own oracle, in random sweep order, with cooled 15 s blocks and full-matrix errors. The audit stopped early under its own rule (native drifted 14-56% inside a block because a 15 s block heats the GPU), so only 3 blocks per family exist and the confidence bounds (t distribution, 2 degrees of freedom) are **indicative only**. Speedup of the a-priori engine, mean / lower 95% bound: vs best dynamic, G 1.67 / 1.37, W 2.60 / 2.21, X 2.81 / 1.72; vs best fixed or offset, G 1.19 / 1.07, W 1.65 / 1.39, X 1.53 / 1.39. The pre-set promotion rule (lower bound above 1.25) was met by the oracle engine in all three families and by the a-priori engine on W and X, not on G. Because the audit's own pairing check failed, no promotion is claimed. Raw: `audit_results.jsonl`.

## 8. Fixed operand (reuse of one encoded matrix)

When A is encoded once and multiplied by q different B matrices (inside scaling off, encode charged): persistent / non-persistent engine time 0.88 (G), 0.94 (X), 0.87 (W), 0.89 (R), 1.23 (S, because disabling inside scaling costs 7 moduli) at q = 1; 0.84 at q = 2; 0.77 at q = 4 (G only). The reuse gain is bounded by the A-side encode share of about 20%; measured 12-23% at q <= 4. Longer streams (q = 8, 16) and other families' q = 2, 4 were invalid under the drift rule or not run, so the test specified in advance (q = 16, at least 10% faster) could **not be evaluated**. First and last products matched native accuracy on the sample; accuracy of every intermediate product is not established. See `persist_results.jsonl`.

## 9. What is not established

- A guaranteed all-entry accuracy contract inside the timed path (only a sampled contract plus an optional full-matrix certificate).
- Performance under sustained thermal load, on other GPUs, other drivers, or at n = 16384 (not run).
- A win at n = 2048, or at n = 8192 for families S and W.
- That the exact measured sources are reproduced by the code in this repository: result records carry the hash of the code version that produced them. The code published here is a cleaned copy (it inlines the input-generation helpers ), so its file hash differs from those recorded; a verification rule for future runs is to commit the code before the run so that hash equals commit.
- The full-matrix reference is the engine's own N = 32 result rather than an independent oracle.
- The vendor library's recommended-bits readout reports what it recommends, not whether it fell back; fallback is visible in time only.

## 10. What comes next

A fused INT8 GEMM with a modular-reduction epilogue (removes about 9 N n^2 bytes of traffic, roughly 2-3 ms at n = 4096); fused conversion kernels (the non-GEMM stages run at about 47% of DRAM peak); a certificate inside the timed path; a hot-state sensitivity run; n = 16384; and the accuracy-matched public shoot-out in [status_and_roadmap.md](status_and_roadmap.md).

## Reproducing

```
python code/ozaki_engine.py report     # table above from results/engine/race_results.jsonl (CPU)
bash results/engine/replay.sh          # re-runs the listed measurement steps (the race itself takes hours) of the PUBLISHED implementation into a fresh folder (NVIDIA GPU, CUDA 13.1); the published code is a cleaned copy, not bit-identical to the code behind the stored records
```

## Data hygiene
Environment logs in the evidence directory were subjected to a byte-exact strip of all text-valued environmental arrays to prevent leakage, preserving numerical metrics only.
