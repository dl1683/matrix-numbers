# A verified map of structured-matrix classes

**Purpose.** Which classes of structured matrices keep a compact description when you multiply, invert, or take Kronecker products of their members, how fast does the description grow, and where are the exact statements, the conditional ones, and the gaps?

**Be clear about what this is.** **No new theorem is claimed.** After a focused literature search every candidate statement turned out to be known or elementary (the closest, the exact dimension of the Monarch product class, is a direct consequence of an appendix of Dao et al. 2022; we give an explicit statement and an independent proof). The value is the map: each cell says what is known, what is exact at tested sizes, what is open, and what the strongest specialist baseline is. Rigor labels: **KNOWN** (cited literature), **PROOF** (written here and attacked), **EMPIRICAL** (exact finite check, rigorous at the tested sizes only), **CONJECTURE**, **SPECULATION**.

Code: [`code/floor_ldr_checks.py`](../code/floor_ldr_checks.py). Data: [`results/structured_map/`](../results/structured_map/).

## 1. How the checks work

The dimension of a product-of-sparse-factors variety equals the generic rank of the parameter Jacobian. A Jacobian rank computed modulo a prime at one integer point is a rigorous *lower* bound for that generic rank; explicit gauge symmetries give an *upper* bound. When they meet, the dimension is exact at that size. Exact rational arithmetic (Python fractions) is used for the certificates. Predictions for the numbers were written down before each suite was run, and a mismatch would have withdrawn the statement, not the prediction. Every suite runs on one thread in seconds.

## 2. Notation

n = m^2 for Monarch, L = log2 n for butterflies, Pi = the transposition permutation of the m x m index grid. Mon_LPiR = {L Pi R : L, R block-diagonal with m blocks of size m x m}; Dao et al.'s own Monarch class is Pi.Mon_LPiR, the same set up to a fixed row permutation that does *not* commute with products, so single-matrix statements transfer but product and inverse statements hold only for Mon_LPiR.

## 3. The map

Cells state the descriptor behaviour; costs are native-descriptor operation counts, and dense-to-class conversion is charged separately.

| Class | Product of two members | Inverse | Kronecker product | Strongest specialist baseline |
|---|---|---|---|---|
| Low displacement rank (Toeplitz/Hankel-like pair) | in class along chained operator pairs, rank <= r1 + r2 (KNOWN; checked, generic rank 4 at n = 6) | same rank with the swapped pair (KNOWN; checked) | rank <= r1 n2 + r2 n1 - r1 r2 for the Kronecker operator pair, attained at r1 = r2 = 1 (n1 + n2 - 1) (PROOF); generic Toeplitz 2 n1 + 2 n2 - 4 (EMPIRICAL only) | FFT convolution; dense GEMM |
| Cauchy-like / Vandermonde-like | as above, nodes must chain | as above | as above | FMM, FFT (the DFT is a well-conditioned special case) |
| Butterfly (L sparse factors) | not in class for n = 4, 8, 16, 32 (exact finite check; general n CONJECTURE); k-fold dimension <= n(kL + 1) (PROOF), equality EMPIRICAL (n <= 16 and n = 32); depth >= (n-1)/L necessary, O(n) sufficient (KNOWN) | lands in the reversed-stride variety (KNOWN) | closed, L = L1 + L2 (PROOF) | FFT; dense GEMM |
| Monarch, Mon_LPiR | not in class for every m >= 2: dim Mon^2 = 3m^3 - 2m^2 (PROOF, a direct consequence of Dao et al. 2022); dim Mon^k <= (k+1)m^3 - k m^2 (PROOF), equality PROOF for k <= 2 and EMPIRICAL at 20 tested (m, k) pairs, CONJECTURE in general | closed for Mon_LPiR (PROOF); Dao's class closed up to moving the fixed Pi (PROOF, later additions) | closed after a fixed reordering, in both conventions; plain lexicographic order fails (PROOF, later additions; checked exactly at four size pairs) | FFT (two-stage Cooley-Tukey); dense GEMM |
| HSS / quasiseparable | in class, order <= r1 + r2, sharp (KNOWN; checked) | in class, same off-diagonal ranks (KNOWN, nullity theorem) | **not closed** in lexicographic order: J_k (x) (I_k + J_k) has order k from order-1 factors (PROOF) | banded solvers, 1-D FMM |
| H / H^2 matrices | block rank grows by a sparsity constant; truncated product is approximate (KNOWN) | approximate H-inverse (KNOWN) | CONJECTURE: not closed with bounded block rank | FMM, multigrid, sparse LU |
| Sparse + low rank | low-rank part rank <= r1 + r2; sparse part multiplies (PROOF, direct expansion) | Woodbury (KNOWN) | CONJECTURE: not closed | sparse LU + Woodbury |

Entries tagged "cited, not re-verified" are literature statements; we did not re-derive them.

## 4. Exact numbers behind the table

- **Monarch single matrices:** a matrix is Monarch iff each of its m^2 slices has rank at most 1, so the dimension is m^2(2m - 1) = 2m^3 - m^2, not the 2 n^(3/2) parameter count (PROOF).
- **Monarch products:** dim Mon^2 = 3m^3 - 2m^2 (the 2m^2 gauge removed from 3m^3 parameters), so Mon^2 is Zariski-dense in the m^4-dimensional space only for m = 2 and has codimension m^2(m-1)(m-2) otherwise. Exact Jacobian ranks at an explicit witness: 16, 63, 160, 325 for m = 2..5 (mod 32003; exactly over Q for m = 2, 3). Growth per multiplication: m^3 - m^2 = n^(3/2) - n new degrees of freedom. Conjecture: dim Mon^k = min(m^4, (k+1)m^3 - k m^2), dense exactly at k = m; the upper bound and the necessity "dense needs k >= m" are proved, and equality is checked at 20 (m, k) pairs (dimension 16, 81, 425, 625, 396, 576, 1116, 1296 at the extra pairs (2,3), (3,4), (5,3), (5,5), (6,1), (6,2), (6,5), (6,6)), with saturation at k = m seen for m = 2..6.
- **Dao's own class:** not product-closed for any m >= 2. Exact ranks of its products are 16, 81, 208 at m = 2, 3, 4 (meeting the gauge bound), against 16, 63, 160 for Mon^2. As a set it is not inverse- or transpose-closed with the leading permutation held fixed (an exact certificate at m = 3 with explicit rational blocks and the offending 2x2 minors, `dao_exact.json`, re-verified from the JSON alone by a separate code path), but this is an artefact of the fixed permutation: moving it to the other side restores closure.
- **Butterflies:** dimension at most n(kL + 1) for a product of k butterflies (diagonal gauges between adjacent factors). Dense is first reached at k = 2, 3, 4 for n = 4, 8, 16 and at k = 7 for n = 32, equal to ceil((n-1)/L); sufficiency for general n is not proved (an O(n) bound is known).
- **Kronecker pairs:** the rank formula for I (x) I under the Kronecker operator pair is n1 + n2 - 1 generically (n1, n2 in 2..6, `ldr_kron_family.json`). Order of J_k (x) (I_k + J_k) is k for k = 2..6 (`qs_kron_exact.json`).

## 5. What the map says about composition

Descriptor growth under repeated multiplication of q members:

- Low displacement rank: rank at most the sum of the ranks, linear in q (PROOF by induction); equality "2q for Toeplitz" was checked only at q = 2.
- HSS: order at most additive, attained in the checked example, not universal.
- Monarch: dimension at most (k+1)m^3 - k m^2, slope n^(3/2) - n per product, but the shape changes; a chain of q Monarch factors applied one after another costs q * 2 n^(3/2) and beats one dense matvec only while q < sqrt(n)/2.
- Butterfly: slope at most nL per factor.
- H-matrices: multiplicative in the sparsity constant unless truncated.

**Closure alone is cheap.** Block-diagonal algebras and circulants are closed under product, inverse and (after a fixed reordering) Kronecker product with zero growth, but they are small commutative or semisimple families. The interesting cell is **expressive and closed**: a class closed under product, inverse and Kronecker with bounded growth that is also not confined to such a small subalgebra (informally; not formalised here). **SPECULATION:** plausibly no class in the map fills it. That cell is the target for a new representation, and nothing here exhibits one.

## 6. Cost honesty

The statements concern descriptor dimension and rank, not runtime. Converting a dense matrix into a class costs O(n^2) or more and dominates single uses (Monarch projection is O(n^(5/2)), which exceeds a dense matvec). No wall-clock claim is made.

## 7. Status of the proofs

The statements were re-derived and re-checked; no new theorem is claimed.

## 8. Next

(a) Prove equality of dim Mon^k for all k by a tangent induction on the alternating chain (k = 2 is the argument of Dao et al.; k >= 3 is open). (b) Test whether a lattice of Sylvester operators (per-axis operators, multilevel Toeplitz) gives bounded growth under Kronecker with a tensor-structured descriptor. (c) Construct a class strictly between Monarch and dense with per-product growth O(n) rather than n^(3/2), closed under inverse, and cost it against Mon^k.

## Reproducing

```
python code/floor_ldr_checks.py r3             # a few seconds
python code/floor_ldr_checks.py monarch        # other suites: butterfly ldr dao m2_witness dao_exact ldr_kron_family qs_kron_exact monarch_extra butterfly_extra
```
