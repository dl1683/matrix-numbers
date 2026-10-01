# CRT INT8 Residue Plane Contract Table

## Overview and Model

This table provides an exact, parameterized accounting of Chinese Remainder Theorem (CRT) INT8 residue plane emulation for matrix multiplication versus native GPU execution paths.

### Baseline Hardware & Parameters (RTX 5090 Laptop)
Source: repository measurements in `docs/engine.md` and `results/engine/`.
- **INT8 Tensor Core Peak**: 141.00 TOPS (dense, INT32 accumulation)
- **FP64 Native**: 0.69 TFLOPS
- **FP32 Native**: 25.00 TFLOPS (*FP32 rate assumed, UNVERIFIED*)
- **TF32 Tensor**: 36.00 TFLOPS
- **FP16 Tensor**: 42.00 TFLOPS
- **FP8 Tensor (E4M3)**: 80.00 TFLOPS
- **Overhead Fraction**: 0.30 (*ASSUMED, replace with measured encode+reduce+reconstruct share*)

### CRT Moduli Selection
Moduli are chosen greedily from 256 downward, enforcing pairwise coprimality (gcd(m_i, m_j) = 1 for all i != j). Moduli range in [2, 256].
- **Total available moduli**: 49
- **First 16 moduli**: `[256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193]`
- **Total cumulative representable bits**: 341.88 bits

### Precision Contract & Formulas
1. **Single GEMM Required Magnitude Bits**:
   B = 2(b + s) + ceil(log2 k) + 1
   where b is kept mantissa bits, s is block exponent spread guard bits, and +1 ensures signed uniqueness.
2. **FP64 Chain Depth q Required Magnitude Bits (b=53)**:
   B_q = q(b + s) + (q - 1)ceil(log2 k) + 1
3. **Residue Planes P**:
   Smallest integer P >= 1 such that bits(P) = sum_{i=1..P} log2(m_i) >= B. If B > total_bits, P = None.
4. **Throughput Ceilings & Net Rate**:
   - Single GEMM: int8_ceiling = INT8_TOPS / P
   - Chain depth q: int8_ceiling = INT8_TOPS / P vs native FP64 rate 0.69 TFLOPS (q matrix products across P planes)
   - Net Rate: net_rate = int8_ceiling x (1 - overhead_fraction)
5. **Decision Rules (Verdicts)**:
   - **WIN**: net_rate >= 1.10 x native_rate (at least 10% speedup over native after 30% overhead)
   - **LOSS**: int8_ceiling < native_rate (cannot win even at zero overhead)
   - **MARGINAL**: int8_ceiling >= native_rate but net_rate < 1.10 x native_rate

---

## How to Read These Tables

- **Guard s**: Additional guard bits reserved for dynamic exponent variations across matrix blocks (s=0 indicates uniform scaling).
- **Inner k**: Matrix contraction dimension (k in {1024, 4096, 16384}).
- **Req Bits B**: Minimum bitwidth required by CRT to reconstruct the dot products exactly without modular overflow.
- **Planes P**: Number of INT8 residue planes (independent INT8 GEMMs) required.
- **Moduli Bits**: Total dynamic range supplied by the P moduli sum_{i=1..P} log2(m_i).
- **INT8 Ceil**: Theoretical throughput ceiling before conversion overheads (effective TFLOPS).
- **Net Rate**: Delivered throughput after accounting for 30% conversion and reconstruction overhead.
- **Native Rate**: Measured baseline throughput on the RTX 5090 Laptop.
- **Verdict**: `WIN`, `MARGINAL`, or `LOSS`.

---

### Contract: FP64 (b=53, Native=0.69 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |          117 |         15 |      117.78 |               9.40 |              6.58 |            0.69 |   9.54x |   WIN   | Single matrix product. |
|         0 |      4096 |          119 |         16 |      125.38 |               8.81 |              6.17 |            0.69 |   8.94x |   WIN   | Single matrix product. |
|         0 |     16384 |          121 |         16 |      125.38 |               8.81 |              6.17 |            0.69 |   8.94x |   WIN   | Single matrix product. |
|         8 |      1024 |          133 |         18 |      140.45 |               7.83 |              5.48 |            0.69 |   7.95x |   WIN   | Single matrix product. |
|         8 |      4096 |          135 |         18 |      140.45 |               7.83 |              5.48 |            0.69 |   7.95x |   WIN   | Single matrix product. |
|         8 |     16384 |          137 |         18 |      140.45 |               7.83 |              5.48 |            0.69 |   7.95x |   WIN   | Single matrix product. |
|        12 |      1024 |          141 |         19 |      147.94 |               7.42 |              5.19 |            0.69 |   7.53x |   WIN   | Single matrix product. |
|        12 |      4096 |          143 |         19 |      147.94 |               7.42 |              5.19 |            0.69 |   7.53x |   WIN   | Single matrix product. |
|        12 |     16384 |          145 |         19 |      147.94 |               7.42 |              5.19 |            0.69 |   7.53x |   WIN   | Single matrix product. |
|        24 |      1024 |          165 |         22 |      170.10 |               6.41 |              4.49 |            0.69 |   6.50x |   WIN   | Single matrix product. |
|        24 |      4096 |          167 |         22 |      170.10 |               6.41 |              4.49 |            0.69 |   6.50x |   WIN   | Single matrix product. |
|        24 |     16384 |          169 |         22 |      170.10 |               6.41 |              4.49 |            0.69 |   6.50x |   WIN   | Single matrix product. |

### Contract: FP32 (b=24, Assumed Native=25.0 TFLOPS, UNVERIFIED)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |           59 |          8 |       63.58 |              17.62 |             12.34 |           25.00 |   0.49x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|         0 |      4096 |           61 |          8 |       63.58 |              17.62 |             12.34 |           25.00 |   0.49x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|         0 |     16384 |           63 |          8 |       63.58 |              17.62 |             12.34 |           25.00 |   0.49x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|         8 |      1024 |           75 |         10 |       79.24 |              14.10 |              9.87 |           25.00 |   0.39x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|         8 |      4096 |           77 |         10 |       79.24 |              14.10 |              9.87 |           25.00 |   0.39x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|         8 |     16384 |           79 |         10 |       79.24 |              14.10 |              9.87 |           25.00 |   0.39x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|        12 |      1024 |           83 |         11 |       87.04 |              12.82 |              8.97 |           25.00 |   0.36x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|        12 |      4096 |           85 |         11 |       87.04 |              12.82 |              8.97 |           25.00 |   0.36x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|        12 |     16384 |           87 |         11 |       87.04 |              12.82 |              8.97 |           25.00 |   0.36x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|        24 |      1024 |          107 |         14 |      110.16 |              10.07 |              7.05 |           25.00 |   0.28x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|        24 |      4096 |          109 |         14 |      110.16 |              10.07 |              7.05 |           25.00 |   0.28x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |
|        24 |     16384 |          111 |         15 |      117.78 |               9.40 |              6.58 |           25.00 |   0.26x |  LOSS   | Single matrix product. FP32 rate assumed, UNVERIFIED. |

### Contract: TF32 (b=11, Native=36.0 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |           33 |          5 |       39.90 |              28.20 |             19.74 |           36.00 |   0.55x |  LOSS   | Single matrix product. |
|         0 |      4096 |           35 |          5 |       39.90 |              28.20 |             19.74 |           36.00 |   0.55x |  LOSS   | Single matrix product. |
|         0 |     16384 |           37 |          5 |       39.90 |              28.20 |             19.74 |           36.00 |   0.55x |  LOSS   | Single matrix product. |
|         8 |      1024 |           49 |          7 |       55.71 |              20.14 |             14.10 |           36.00 |   0.39x |  LOSS   | Single matrix product. |
|         8 |      4096 |           51 |          7 |       55.71 |              20.14 |             14.10 |           36.00 |   0.39x |  LOSS   | Single matrix product. |
|         8 |     16384 |           53 |          7 |       55.71 |              20.14 |             14.10 |           36.00 |   0.39x |  LOSS   | Single matrix product. |
|        12 |      1024 |           57 |          8 |       63.58 |              17.62 |             12.34 |           36.00 |   0.34x |  LOSS   | Single matrix product. |
|        12 |      4096 |           59 |          8 |       63.58 |              17.62 |             12.34 |           36.00 |   0.34x |  LOSS   | Single matrix product. |
|        12 |     16384 |           61 |          8 |       63.58 |              17.62 |             12.34 |           36.00 |   0.34x |  LOSS   | Single matrix product. |
|        24 |      1024 |           81 |         11 |       87.04 |              12.82 |              8.97 |           36.00 |   0.25x |  LOSS   | Single matrix product. |
|        24 |      4096 |           83 |         11 |       87.04 |              12.82 |              8.97 |           36.00 |   0.25x |  LOSS   | Single matrix product. |
|        24 |     16384 |           85 |         11 |       87.04 |              12.82 |              8.97 |           36.00 |   0.25x |  LOSS   | Single matrix product. |

### Contract: FP16 (b=11, Native=42.0 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |           33 |          5 |       39.90 |              28.20 |             19.74 |           42.00 |   0.47x |  LOSS   | Single matrix product. |
|         0 |      4096 |           35 |          5 |       39.90 |              28.20 |             19.74 |           42.00 |   0.47x |  LOSS   | Single matrix product. |
|         0 |     16384 |           37 |          5 |       39.90 |              28.20 |             19.74 |           42.00 |   0.47x |  LOSS   | Single matrix product. |
|         8 |      1024 |           49 |          7 |       55.71 |              20.14 |             14.10 |           42.00 |   0.34x |  LOSS   | Single matrix product. |
|         8 |      4096 |           51 |          7 |       55.71 |              20.14 |             14.10 |           42.00 |   0.34x |  LOSS   | Single matrix product. |
|         8 |     16384 |           53 |          7 |       55.71 |              20.14 |             14.10 |           42.00 |   0.34x |  LOSS   | Single matrix product. |
|        12 |      1024 |           57 |          8 |       63.58 |              17.62 |             12.34 |           42.00 |   0.29x |  LOSS   | Single matrix product. |
|        12 |      4096 |           59 |          8 |       63.58 |              17.62 |             12.34 |           42.00 |   0.29x |  LOSS   | Single matrix product. |
|        12 |     16384 |           61 |          8 |       63.58 |              17.62 |             12.34 |           42.00 |   0.29x |  LOSS   | Single matrix product. |
|        24 |      1024 |           81 |         11 |       87.04 |              12.82 |              8.97 |           42.00 |   0.21x |  LOSS   | Single matrix product. |
|        24 |      4096 |           83 |         11 |       87.04 |              12.82 |              8.97 |           42.00 |   0.21x |  LOSS   | Single matrix product. |
|        24 |     16384 |           85 |         11 |       87.04 |              12.82 |              8.97 |           42.00 |   0.21x |  LOSS   | Single matrix product. |

### Contract: FP8-E4M3 (b=4, Native=80.0 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |           19 |          3 |       23.98 |              47.00 |             32.90 |           80.00 |   0.41x |  LOSS   | Single matrix product. |
|         0 |      4096 |           21 |          3 |       23.98 |              47.00 |             32.90 |           80.00 |   0.41x |  LOSS   | Single matrix product. |
|         0 |     16384 |           23 |          3 |       23.98 |              47.00 |             32.90 |           80.00 |   0.41x |  LOSS   | Single matrix product. |
|         8 |      1024 |           35 |          5 |       39.90 |              28.20 |             19.74 |           80.00 |   0.25x |  LOSS   | Single matrix product. |
|         8 |      4096 |           37 |          5 |       39.90 |              28.20 |             19.74 |           80.00 |   0.25x |  LOSS   | Single matrix product. |
|         8 |     16384 |           39 |          5 |       39.90 |              28.20 |             19.74 |           80.00 |   0.25x |  LOSS   | Single matrix product. |
|        12 |      1024 |           43 |          6 |       47.81 |              23.50 |             16.45 |           80.00 |   0.21x |  LOSS   | Single matrix product. |
|        12 |      4096 |           45 |          6 |       47.81 |              23.50 |             16.45 |           80.00 |   0.21x |  LOSS   | Single matrix product. |
|        12 |     16384 |           47 |          6 |       47.81 |              23.50 |             16.45 |           80.00 |   0.21x |  LOSS   | Single matrix product. |
|        24 |      1024 |           67 |          9 |       71.41 |              15.67 |             10.97 |           80.00 |   0.14x |  LOSS   | Single matrix product. |
|        24 |      4096 |           69 |          9 |       71.41 |              15.67 |             10.97 |           80.00 |   0.14x |  LOSS   | Single matrix product. |
|        24 |     16384 |           71 |          9 |       71.41 |              15.67 |             10.97 |           80.00 |   0.14x |  LOSS   | Single matrix product. |

### Contract: FP64 Chain Depth q=2 (b=53, Native=0.69 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |          117 |         15 |      117.78 |               9.40 |              6.58 |            0.69 |   9.54x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 15 residue planes). |
|         0 |      4096 |          119 |         16 |      125.38 |               8.81 |              6.17 |            0.69 |   8.94x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 16 residue planes). |
|         0 |     16384 |          121 |         16 |      125.38 |               8.81 |              6.17 |            0.69 |   8.94x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 16 residue planes). |
|         8 |      1024 |          133 |         18 |      140.45 |               7.83 |              5.48 |            0.69 |   7.95x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 18 residue planes). |
|         8 |      4096 |          135 |         18 |      140.45 |               7.83 |              5.48 |            0.69 |   7.95x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 18 residue planes). |
|         8 |     16384 |          137 |         18 |      140.45 |               7.83 |              5.48 |            0.69 |   7.95x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 18 residue planes). |
|        12 |      1024 |          141 |         19 |      147.94 |               7.42 |              5.19 |            0.69 |   7.53x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 19 residue planes). |
|        12 |      4096 |          143 |         19 |      147.94 |               7.42 |              5.19 |            0.69 |   7.53x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 19 residue planes). |
|        12 |     16384 |          145 |         19 |      147.94 |               7.42 |              5.19 |            0.69 |   7.53x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 19 residue planes). |
|        24 |      1024 |          165 |         22 |      170.10 |               6.41 |              4.49 |            0.69 |   6.50x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 22 residue planes). |
|        24 |      4096 |          167 |         22 |      170.10 |               6.41 |              4.49 |            0.69 |   6.50x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 22 residue planes). |
|        24 |     16384 |          169 |         22 |      170.10 |               6.41 |              4.49 |            0.69 |   6.50x |   WIN   | Chain depth q=2: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (2 factors = 1 matrix products, each across 22 residue planes). |

### Contract: FP64 Chain Depth q=3 (b=53, Native=0.69 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |          180 |         24 |      184.64 |               5.88 |              4.11 |            0.69 |   5.96x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 24 residue planes). |
|         0 |      4096 |          184 |         24 |      184.64 |               5.88 |              4.11 |            0.69 |   5.96x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 24 residue planes). |
|         0 |     16384 |          188 |         25 |      191.86 |               5.64 |              3.95 |            0.69 |   5.72x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 25 residue planes). |
|         8 |      1024 |          204 |         27 |      206.07 |               5.22 |              3.66 |            0.69 |   5.30x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 27 residue planes). |
|         8 |      4096 |          208 |         28 |      213.11 |               5.04 |              3.52 |            0.69 |   5.11x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 28 residue planes). |
|         8 |     16384 |          212 |         28 |      213.11 |               5.04 |              3.52 |            0.69 |   5.11x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 28 residue planes). |
|        12 |      1024 |          216 |         29 |      220.10 |               4.86 |              3.40 |            0.69 |   4.93x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 29 residue planes). |
|        12 |      4096 |          220 |         29 |      220.10 |               4.86 |              3.40 |            0.69 |   4.93x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 29 residue planes). |
|        12 |     16384 |          224 |         30 |      226.92 |               4.70 |              3.29 |            0.69 |   4.77x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 30 residue planes). |
|        24 |      1024 |          252 |         34 |      253.77 |               4.15 |              2.90 |            0.69 |   4.21x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 34 residue planes). |
|        24 |      4096 |          256 |         35 |      260.37 |               4.03 |              2.82 |            0.69 |   4.09x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 35 residue planes). |
|        24 |     16384 |          260 |         35 |      260.37 |               4.03 |              2.82 |            0.69 |   4.09x |   WIN   | Chain depth q=3: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (3 factors = 2 matrix products, each across 35 residue planes). |

### Contract: FP64 Chain Depth q=4 (b=53, Native=0.69 TFLOPS)

| Guard s | Inner k | Req Bits B | Planes P | Moduli Bits | INT8 Ceil (TFLOPS) | Net Rate (TFLOPS) | Native (TFLOPS) | Speedup | Verdict | Notes |
|----------:|----------:|-------------:|-----------:|------------:|-------------------:|------------------:|----------------:|--------:|:-------:|:------|
|         0 |      1024 |          243 |         33 |      247.11 |               4.27 |              2.99 |            0.69 |   4.33x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 33 residue planes). |
|         0 |      4096 |          249 |         34 |      253.77 |               4.15 |              2.90 |            0.69 |   4.21x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 34 residue planes). |
|         0 |     16384 |          255 |         35 |      260.37 |               4.03 |              2.82 |            0.69 |   4.09x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 35 residue planes). |
|         8 |      1024 |          275 |         38 |      279.52 |               3.71 |              2.60 |            0.69 |   3.76x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 38 residue planes). |
|         8 |      4096 |          281 |         39 |      285.71 |               3.62 |              2.53 |            0.69 |   3.67x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 39 residue planes). |
|         8 |     16384 |          287 |         40 |      291.86 |               3.52 |              2.47 |            0.69 |   3.58x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 40 residue planes). |
|        12 |      1024 |          291 |         40 |      291.86 |               3.52 |              2.47 |            0.69 |   3.58x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 40 residue planes). |
|        12 |      4096 |          297 |         41 |      297.93 |               3.44 |              2.41 |            0.69 |   3.49x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 41 residue planes). |
|        12 |     16384 |          303 |         42 |      303.86 |               3.36 |              2.35 |            0.69 |   3.41x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 42 residue planes). |
|        24 |      1024 |          339 |         49 |      341.88 |               2.88 |              2.01 |            0.69 |   2.92x |   WIN   | Chain depth q=4: effective ceiling = INT8_TOPS / P_chain vs 0.69 native rate (4 factors = 3 matrix products, each across 49 residue planes). |
|        24 |      4096 |          345 |       None |         N/A |                N/A |               N/A |            0.69 |   0.00x |  LOSS   | Exceeds capacity: required 345 bits > all 49 moduli (341.88 bits). Cannot emulate with moduli <= 256. |
|        24 |     16384 |          351 |       None |         N/A |                N/A |               N/A |            0.69 |   0.00x |  LOSS   | Exceeds capacity: required 351 bits > all 49 moduli (341.88 bits). Cannot emulate with moduli <= 256. |

---

## Synthesis and Architectural Findings

1. **FP64 on slow-FP64 hardware**: On this laptop part native FP64 runs at 1/64 of FP32 throughput (0.69 TFLOPS). The model gives an INT8 CRT ceiling of 6.41 to 9.40 TFLOPS and a net rate of 4.49 to 6.58 TFLOPS (6.5x to 9.5x net speedup) for guard bits s up to 24. This is a planning model with an assumed 30% overhead and a sustained-load INT8 rate; measured results are in docs/engine.md.
2. **FP32, TF32, FP16 and FP8 are losses for CRT emulation**: native tensor-core paths for TF32 (36 TFLOPS), FP16 (42 TFLOPS) and FP8 (80 TFLOPS) are already fast. Even at minimal bit requirements (B=19..37, P=3..5 planes), the maximum INT8 CRT throughput (141/P = 28.2 to 47.0 TFLOPS) stays below the native hardware rates before any overhead. CRT emulation targets precisions the hardware is slow at, not precisions it is fast at.
3. **FP64 chains**: multi-product chains (A_1 ... A_q) can keep intermediate products in residue representation without full CRT decoding until the final step. For q=2, 3 and 4 factors the model gives a WIN in every feasible row (net speedup over the same number of native products: 6.5x to 9.5x, 4.1x to 6.0x and 2.9x to 4.3x). At q=4 with s=24 and k >= 4096 the 8-bit moduli run out (required bits exceed the 341.88 available).

