# Vision: primitives and the shape of computing

*This page states a thesis. It is not a result. Nothing here has been shown by the measurements in this repository; the results are in the other documents, with their own rigor labels.*

## The observation

Every era of computing has been defined by its primitives. The Turing machine fixed one: a tape of symbols, one cell read at a time, and a notion of "step" that made a theory of computation possible. Our number systems did the same thing earlier and more quietly. The real line is built around 1. Operations act entry by entry. Anything higher-dimensional (a vector, a matrix, a tensor) is an array of scalars, an extension bolted on top of the primitive rather than the primitive itself.

Meanwhile, two parts of the stack have already moved. Physics made the swap a century ago: in matrix mechanics the observables are operators, and scalars only appear at measurement. Hardware made it more recently: a tensor core's instruction is a small matrix product, and a TPU's native operation is a 128x128 multiply. Mathematical notation and the arithmetic we teach machines to emulate are the layers in the middle that still think in scalars.

## The thesis

We suspect that the primitives we inherited (scalars, entry-wise arithmetic, dense arrays) may be quietly limiting how efficiently high-dimensional computation can be done, and that primitives native to structured, high-dimensional objects could change what "efficient" means. If so, computation built for the kinds of objects now dominating scientific and learning workloads might be best served by an arithmetic whose atom is the matrix and whose scalar is the 1x1 corner case, rather than by better kernels for the old atom.

This is a hypothesis about where the leverage is. It could be wrong. The honest alternatives are that existing structured algorithms (FFTs, block structure, specialised factorisations) already capture what a matrix-native arithmetic could, or that the gains live entirely in conversion and precision costs that no representation can avoid.

## How the repository tests it

We do not argue the thesis; we try to make it fail at small, measurable scales.

- **Build:** a matrix-native residue representation of FP64 matrix products on integer tensor cores ([engine](engine.md)). It shows that changing the representation changes the cost: 16-20 small-integer products replace one slow double-precision product, with conversion, workspace and accuracy all charged.
- **Count:** exact lower bounds and certified optimal cases for how many low-precision products a higher-precision product needs ([theory](product_count_theory.md)), including a cost law that says where a persistent residue representation does not help.
- **Map:** which structured-matrix classes keep a compact description under multiplication, inversion and Kronecker products ([map](structured_matrix_map.md)), and which cell of that map remains empty. That empty cell, a class that is expressive and closed under everything with bounded growth, is the kind of object the thesis would predict and nothing here exhibits.

## Implications, kept separate

The work is general-purpose mathematics and computation: analytical structure, exact arithmetic, accuracy contracts. If the thesis has substance, its implications reach any field limited by large matrix products, including high-dimensional workloads such as those in machine learning. Those are implications to be earned by results in the layers above, not the definition of the project, and no result in this repository concerns a learning workload.

## What would change our mind

A measured accuracy-matched shoot-out in which the strongest specialised method (a vendor emulation mode, an FFT, a symmetry-adapted code) beats the matrix-native route once conversion, memory traffic and precision are charged; or a proof that every useful representation in a stated class pays its savings back in conversion. Both outcomes would be recorded here with the same weight as a win.
