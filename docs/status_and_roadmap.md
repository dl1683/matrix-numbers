> **Research moved to [Latent Space Reasoning](https://github.com/dl1683/Latent-Space-Reasoning).**
> All future work from this project continues in LSR, alongside persistent latent
> memory, long-context computation, new model construction/training/inference,
> and shared high-dimensional and neural mathematics. This repository preserves
> historical research; its old run/resume instructions are historical, not a
> current launch queue. Start with [LSR's current state](https://github.com/dl1683/Latent-Space-Reasoning/blob/main/STATE.md)
> and [unified evidence](https://github.com/dl1683/Latent-Space-Reasoning/blob/main/docs/UNIFIED_EVIDENCE.md).
> Consolidation does not expand the scientific scope of the results below.

# Status and roadmap

An honest summary of where the work stands. Labels: PROVED / EMPIRICAL / CONJECTURE / SPECULATION, as in the other documents.

## Established (with the strength it deserves)

| Result | Label | Strength and limits | Document |
|---|---|---|---|
| FP64 matrix product from 16-20 INT8 residue products: about 11-17x over native FP64, about 1.5-3x over the vendor's dynamic emulation, about 1.2-1.7x over a calibrated vendor variant (fixed mantissa or dynamic offset), at n >= 4096 | EMPIRICAL | one laptop GPU; cool-start; sampled accuracy contract; 13 of 15 race cells valid (latest record per cell); n = 8192 under amended split brackets; independently re-audited from raw records | [engine](engine.md) |
| Per-entry error bound for the engine (Lemma L1), data-independent configuration table | proof written, independent re-check pending; 0 violations in all audits | the all-entry bound holds; the choice of the number of moduli is statistical | [engine](engine.md) |
| Eleven published low-precision product counts are optimal in the bilinear model | theorems + EXACT-COMPUTATION | independent proof re-check pending | [theory](product_count_theory.md) |
| 32 FP8 products instead of 36 for FP64-level products | EMPIRICAL-exact | pure-FP8 hardware only; 16 CRT terms instead of 12; nothing timed | [theory](product_count_theory.md) |
| 16 INT8 products is optimal for exact 54-bit significand products with one product per modulus | EXACT-COMPUTATION | the general floor is 13; 13 vs 16 is open | [theory](product_count_theory.md) |
| Cost law for chains of products in residue form (persistent residues save bytes, not products) | PROVED (elementary) + EXACT-COMPUTATION | a statement about counts and bytes, not a measurement | [theory](product_count_theory.md) |
| A verified map of structured-matrix classes under product, inverse and Kronecker | KNOWN + PROOF + EMPIRICAL | no new theorem claimed; later additions not independently re-attacked | [map](structured_matrix_map.md) |

## In progress

- **INT8 x INT8 = 6 (the six-product scheme is optimal).** An exhaustive search with a coupling test; the complete search list is still being searched, so the claim is conditional and is not counted among the eleven.
- **Certificate inside the timed path.** The full-matrix certificate exists and passes at n = 2048, but costs more than the engine itself unoptimised; an optimised fused version is bounded below by one FP16 product, about 10-12% of the engine time.
- **Fused kernels.** A fused INT8 product with a modular-reduction epilogue (about 9 N n^2 bytes less traffic) and fused conversion kernels (the non-product stages run at about 47% of DRAM peak).
- **The 13-versus-16 gap** for exact FP64 significands: an inverse Littlewood-Offord route is identified, not carried out.
- **Open cells** (5,8), (7,8), (4,9), (5,9) of the product-count table, and the 2-product prime-power moduli.
- **Fixed-operand reuse.** Gains of 12-23% at q <= 4 were measured; the long-stream kill test could not be evaluated under the drift rule.

## Not established

- Any win at n = 2048 or at n = 8192 for two input families; any sustained-load or other-GPU result; n = 16384.
- A guaranteed all-entry accuracy contract in the timed path.
- That matrix-native representations beat entry-wise ones in general. The engine shows that a changed representation lowers the product count and the measured time on one class of hardware; it is not a new exponent and needs no new arithmetic.
- Any machine-learning result. The work is general mathematics; high-dimensional computing is discussed only as an implication.

## Next benchmarks

1. **An accuracy-matched shoot-out, open and reproducible.** The engine against every vendor emulation mode and against native FP64, at several sizes and input families, including the families where vendor modes fall back, with accuracy matched by an independent correctly-rounded reference and all of time, bytes and error reported. Interleaved timing with a drift rule, hot and cool regimes both reported.
2. **A public scientific kernel.** One real workload where FP64 matrix products dominate and accuracy can be checked against a published reference; density-fitted exchange in quantum chemistry is the leading candidate. The point is to show the engine inside a real calculation with its real accuracy requirement, not on random matrices.
3. **LAPACK-style residual tests** (backward-error tests of factorisations that call the engine), and the graded-exponent and rotated test families of the BLAS accuracy-grading paper.

## A note on official rankings

The official TOP500 and Green500 lists exclude integer-emulated arithmetic, so a result built from INT8 products cannot appear on them regardless of its accuracy. We therefore publish an **open corpus with raw records for independent recomputation** instead: raw per-round records, the replay script, the input generators with seeds, the accuracy reference, and the baselines, so that anyone can re-run the comparison and attack it.

## Roadmap for the mathematics

- Close the (8,8) and (7,8) cells and decide whether any scheme with 13-15 products exists for exact FP64 significands.
- Models that escape the bilinear exclusions: bounded-inner-dimension post-processing, block-scaled formats, mixed FP8/FP4 planes, 2-product prime-power moduli.
- The structured-matrix map's empty cell: a class that is expressive and closed under product, inverse and Kronecker with bounded descriptor growth, costed against the strongest specialist method including conversion.
- Price every representation in time, bytes and error. A speedup that hides its cost in the encoding step or in precision is not a speedup.
