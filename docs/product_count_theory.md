# How many low-precision products does one exact higher-precision product need?

**Question.** For a hardware value set V (what one operand entry of a low-precision matrix product can hold) and an accumulator that is exact under a stated bound, what is the least number r of low-precision products whose outputs determine the exact product of two integer matrices (or of the truncated significands used by an FP64 emulation)?

Code: [`code/digit_gemm.py`](../code/digit_gemm.py). Data: [`results/theory/`](../results/theory/). Calculator: [`code/contract_plane_table.py`](../code/contract_plane_table.py), [`results/calculator/`](../results/calculator/).

**Rigor labels in this document.** *Theorems* below have written proofs that were written and the finite statements are machine-checked; an independent proof re-check is pending, so they are labelled **PROOF (independent re-check pending)**, never PROVED. *Exhaustive decisions* are **EXACT-COMPUTATION**: they rest on those theorems and on exact (integer or rational) arithmetic, and the screened-in candidates are re-verified exactly. Counts quoted from papers are **LITERATURE**. Nothing in this document is timed.

## 1. Prior art and gap

- Hayashi, Mukunoki, Hoshino, Katagiri (arXiv 2609.24519, called AWE below): search-found counts for emulating integer products on FP4 hardware (INT8 x INT8 in 6 products, INT4 x INT8 in 4, INT4 x INT4 in 3, and a full table over widths). No lower bound was proved. Their search for 5 products covered only planes with small components; they state the question as open. Also Hayashi et al. (arXiv 2608.06812), base-13 limbs, 9 products for INT8 x INT8, no optimality claim.
- Classical bilinear complexity: Toom, Karatsuba, Winograd (1976), Fiduccia and Zalcstein (1977), Brockett and Dobkin (1978), Buergisser, Clausen and Shokrollahi (1997): rank 2s-1 of polynomial multiplication and code connections, all with free coefficients. They do not treat a value set with holes on the planes.
- Ozaki-I and Ozaki-II (Ozaki, Ogita, Oishi, Rump 2012; Ootomo, Ozaki, Yokota; Ozaki, Uchino, Imamura arXiv 2504.08009; Uchino, Ozaki, Imamura arXiv 2603.10634 for FP8 with 3 products per modulus): engineering counts, 14-16 INT8 products for FP64-level accuracy, 36 FP8 products. They give no per-modulus optimality or lower bounds.

## 2. Models

A scheme applies entrywise encodings P_k, Q_k into V, computes r exact products G_k = P_k(A) Q_k(B), and post-processes.

- **M1**: C = sum lambda_k G_k, valid for every inner dimension (AWE's model is a special case).
- **M2**: the same congruence modulo M, valid while 2 max|C| < M (M2p: M prime; M2crt: M a product of pairwise-coprime moduli, one product per modulus, i.e. Ozaki-II).
- **M3 / M4**: arbitrary post-processing, valid for every n (M3) or for n <= N only (M4).
- **Mpoly**: classical control, planes linear in limbs, target all coefficients of the digit convolution.

Value sets: V_FP4 = 2 x E2M1 = {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12} (15 values, FP32 accumulation); V_FP8 = E4M3 integers in [-16, 16] (33 values); V_INT8 = [-128, 127] (INT32 accumulation); V_INT4 = [-8, 7]. Inputs of width M have the symmetric range [-(2^(M-1)-1), 2^(M-1)-1].

## 3. Lower-bound theory (PROOF, independent re-check pending)

- **Theorem 2 (value-set Sylvester bound).** In any M1 scheme r >= rho_A + rho_B - 1 with rho >= ell(V, X), where ell is the least number of V-coordinates whose linear combinations cover the input range X, and ell >= ceil(log_|V| |X|). Proof: the reconstruction form restricted to the spans of the plane vectors has rank 1; apply Sylvester's rank inequality.
- **Theorem 3 (the tight case is MDS and twisted-dual).** If r = ell_A + ell_B - 1 then both plane codes are MDS codes and their extended codes are dual up to a diagonal twist; for equal ranks s the count 2s-1 is attained only by generalised Toom schemes.
- **Lemma 4 (ratio lemmas).** In a tight scheme with enough inputs, every minimum-weight relation has coefficient ratios in (V-V)/(V-V), which makes the search finite.
- **Lemma 5.** Arbitrary post-processing valid for every n reduces to M1 with zero-point corrections: nonlinear post-processing buys nothing at matrix-product scale; only bounded-n tricks or modular reduction can escape the M1 bounds.
- **Theorem 6 (per-modulus bounds).** Two products per prime modulus are impossible for every value set (r = 1 iff V covers Z/p, otherwise r >= 3). Every modulus in the FP8 list of arXiv 2603.10634 needs at least 3 products, so its per-modulus count is optimal and any saving must come from modulus selection. 2-product prime-power moduli are excluded except for a short open list (FP8: 64, 128, 256, 512, 81, 243, 125, 49).

## 4. Results

### 4.1 Eleven published counts are optimal in their model (EXACT-COMPUTATION)

AWE's INT8 x INT8 six-product scheme and INT4 x INT8 four-product scheme were reproduced exactly (identity checked over all 875 x 875 admissible pairs; `awe_reproduction.json`). For the FP4 model M1, the lower bound equals the published count at these width pairs, so the published scheme is optimal there:

| widths (M_A, M_B) | count | how decided |
|---|---|---|
| (4,4), (4,5), (5,5), (4,8) = 3, 3, 3, 4 | standard bound alone (Theorem 2 with the counting / progression bound) | new relative to the literature (which states no bound), not a strengthening of the bound |
| (4,7), (5,7), (7,7) = 4, 4, 5 | exact ell(V_FP4, INT7) = 3, strictly above the counting bound | exhaustive: covering 127 values with two digits forces denominator in {1,2}; complete search finds no witness |
| (4,6), (5,6), (6,6) = 4 | tight-case exhaustion at INT6 | 29 admissible plane codes, none carries a covering functional; controls: the same search finds 72 covering functionals for INT5 on the Karatsuba code and reproduces brute force on all 29 INT6 codes at a reduced target |
| (6,7) = 5 | monotonicity: INT6 has no tight partner of any kind | one-line proof from the sub-code argument |

That is 4 + 3 + 3 + 1 = 11 cells (`ell_and_bound_table.json`, `int6_controls.json`, `cover_controls.json`, `lines_controls.json`).

**The flagship, INT8 x INT8 = 6, is conditional and not claimed.** Any 5-product scheme would have both plane codes MDS [5,3] with at least 255 points. Enumeration found 5,678 orbits; 28 of them carry covering functionals (457 up to sign, denominators 1, 2, 3, 4, 6, 8, 12), found by an independent reimplementation and cross-checked by the search kernel. A 5-product scheme needs both sides covered and coupled by Theorem 3(iii); a pairing test against all 120 coordinate orderings of all 457 extended codes returned 0 matches (positive control: 5 covered codes, randomly permuted and rescaled, each match themselves). A cross-check on seven covered orbits found one more functional than the independent implementation had, so its covered list is a lower bound and the pairing test is being re-run on the complete search list. Until that finishes the verdict for (8,8) is conditional. Cells (5,8), (7,8), (4,9) and (5,9) remain open.

### 4.2 Pure-FP8 hardware: 32 products instead of 36 (EMPIRICAL-exact, scope-limited)

For FP64-level products (V = E4M3 integers with |v| <= 16, inner dimension up to 65,536, modulus product above 2^111) the published fast mode uses 36 FP8 products (12 moduli x 3). Mixing 1-product moduli (m <= 33, where V covers Z/m) with 3-product moduli certified by a 2-limb Karatsuba recipe, a mixed-integer program over pairwise-coprime choices gives **32 = 8 x 3 + 8 x 1**: moduli {769, 767, 751, 749, 733, 719, 709, 701} at 3 products each and {33, 32, 31, 29, 25, 23, 19, 17} at 1 each, log2 of the modulus product 113.56. Every per-modulus identity was checked over all residue pairs, and an end-to-end run (`fp8_e2e.json`: random integer matrices 8x256x8 at 50 bits, 16x1024x16 at 48 bits, 8x65536x8 at 40 bits) reproduces the exact integer product in every entry. Restricting to 3-product moduli reproduces the published 36.

**Caveats.** Optimality is claimed only within the declared candidate family. The cost is 16 CRT terms instead of 12 (more reconstruction multiply-adds). No wall-clock is measured. The result applies to pure-FP8 hardware; where INT8 exists, INT8 CRT needs 15-16 products. The mechanism (mixed per-modulus costs) is AWE's FP4 idea transferred to FP8. The same program for FP4 gives 62, worse than AWE's 59, so no FP4 improvement exists in our candidate family.

### 4.3 FP64 via INT8: the exact-product bound is 16 with one product per modulus

For the exact product of 54-bit signed significands at inner dimension 4096:

| bound | value | status |
|---|---|---|
| M1 / M2p floor (Theorem 2) | 13 | PROOF (independent re-check pending) |
| M4 floors (any post-processing, bounded n) | 7 by injectivity, 5 by output information | proved, weak |
| INT8 CRT with one product per modulus | **16 is optimal** | EXACT-COMPUTATION: the program ranges over every pairwise-coprime subset of 2..256; 15 moduli give at most 118.00 bits against the 119.0 needed |
| below 16 | needs a 2-product modulus (prime powers p^e with p^2 <= 256, undecided) or an M1 scheme with 13-15 products | OPEN |

A 13-product scheme would have to be a tight generalised-Toom scheme with both plane codes MDS [13,7] over INT8. There is a heuristic obstruction (anticoncentration of a sum over a value box makes six dependent planes hard to keep within range) that is **not a proof**. The gap 13 versus 16 is open with a stated route (an inverse Littlewood-Offord bound over the value box).

Engine coupling: `results/theory/config_table.json` lists, for b-bit scaled inputs and inner dimension k, the certified moduli set and count: 4 (b = 8), 6 (b = 16), 8 (b = 24), 10 (b = 32), 14-15 (b = 48), 16 (b = 53) at k in {4096, 16384}, against the floor 3, 5, 7, 9, 13, 13. These counts are within 2-3 products of the floor for b >= 24. The engine ([engine.md](engine.md)) uses the 16-20-modulus end of this table.

### 4.4 Chains of products in residue form: the cost law

For an exact chain A_1 ... A_q with entries at most H = 2^b - 1 and inner dimension k, the range after s factors is R_s = H^s k^(s-1) (in general the product of the entry bounds times the product of the inner dimensions), and each product in residue form costs one INT8 product per modulus (EXACT-COMPUTATION by integer program; the model is elementary and proved).

- Fixed base: (q-1) N(R_q) products. Growing base (add moduli by base extension before each product; elementwise work, no extra products): the sum over s of N(R_s), about 60% of the fixed base.
- Hard cap: the product of all pairwise-coprime moduli up to 256 is lcm(1..256) = 2^362.8, so exact INT8 residue chains end at log2(2 R_q) <= 362.8: at most q = 19 for 7-bit data, 13 for 16-bit data, 5 for 53-bit data at k = 4096.
- Against rounding every intermediate back to b bits, persistent exactness costs a factor of about (q+1)/2 in products (b = 16, q = 8: 119 versus 42; b = 53, q = 4: 74 versus 48).

**So what.** Persistent residues never save products. They buy exactness and remove per-step reconstruction and re-encoding, which is measured in bytes moved, the only currency in which they can win. A residue engine must therefore be priced in bytes, not in product count. (`chain_cost_model.json`; `python code/digit_gemm.py chain`)

### 4.5 The calculator: accuracy contract to residue planes and throughput ceiling

`python code/contract_plane_table.py` (selftest: `--selftest`) maps a precision contract to the number of INT8 residue planes and a throughput ceiling. For b kept mantissa bits, s guard bits for exponent spread and inner dimension k, the required magnitude is B = 2(b + s) + ceil(log2 k) + 1 bits, and P is the smallest number of greedily chosen pairwise-coprime moduli up to 256 whose logarithms sum to at least B (49 moduli, 341.88 bits in total). The ceiling is INT8 TOPS / P, the net rate subtracts an **assumed** 30% overhead, and a verdict compares it with the native rate.

For FP64 (b = 53) at k = 1024 to 16384 the model gives P = 15-16 planes at s = 0, 18 at s = 8, 19 at s = 12, 22 at s = 24, with a net-speedup range of 6.5x to 9.5x over a native rate of 0.69 TFLOPS. For TF32, FP16 and FP8 the model says the emulation loses to the native tensor-core path (ceiling below the native rate). The FP32 native rate (25 TFLOPS) is an assumption and labelled UNVERIFIED in the table. The INT8 rate (141 TOPS) is the sustained-load cuBLASLt rate measured at n = 4096; the cool-start race of [engine.md](engine.md) ran faster than the model's net rate, so the calculator is a conservative planning tool, not a prediction. Full tables: [`results/calculator/TABLE.md`](../results/calculator/TABLE.md), data in `table.json`.

### 4.6 Classical control (Mpoly)

When the post-processing must recover every digit-convolution coefficient, the target has bilinear rank 2s-1, attained by evaluation at 2s-1 points. With V_FP4 and a full digit range every admissible plane is a multiple of one limb, so r = s^2; with a reduced digit range Karatsuba applies for s = 2 (3 products) and not Toom-3 (5 is impossible for s = 3; 6 with 3-term Karatsuba). This is the classical side of AWE's remark about minimum significand widths. It is a control, not the right model for integer emulation (`mpoly.json`).

## 5. What is not established

- The independent proof re-check of Theorems 2, 3, 6 and Lemma 4, and of the pairing test, is pending.
- INT8 x INT8 = 6 (conditional), and cells (5,8), (7,8), (4,9), (5,9) are open.
- The 13-versus-16 gap for exact FP64 significands and the 2-product prime-power moduli are open.
- No wall-clock claim is made by this theory; costs are product counts and bytes.
- Escapes from M1 that the engine could use (bounded-n post-processing, block-scaled formats, mixed FP8/FP4 planes) are not decided.

## 6. So what

Low-precision emulation counts are now partly a closed theory: the bilinear floor, its tight case (generalised Toom), and the value-set ratio lemmas decide small cells exactly and confirm that the search-found counts are optimal where decided. The one constructive gain (32 below 36 on FP8) comes from modulus selection, not from cleverer per-modulus algebra, because the per-modulus counts are already optimal.

## Reproducing

```
python code/digit_gemm.py awe                 # AWE's six-product scheme, exactly (seconds)
python code/digit_gemm.py chain               # chain cost model
python code/digit_gemm.py e2e_fp8             # exact end-to-end FP8 check
python code/digit_gemm.py two_product_moduli  # two-product modulus exclusions
python code/contract_plane_table.py --selftest
```
The large exhaustive searches (candidate rows, orbit enumeration, covering functionals) are in `digit_gemm.py` as further subcommands; their inputs and outputs are too large to store here and take hours.
