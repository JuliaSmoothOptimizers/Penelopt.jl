# Penelopt.jl: A Large-Scale Equality-Constrained Optimization Solver 

| **License** | **Documentation** | **CI** | **Coverage** | **Contributors** |
|:-----------:|:-----------------:|:------:|:------------:|:----------------:|
| [![license-img]][license-url] | [![docs-stable-img]][docs-stable-url] [![docs-dev-img]][docs-dev-url] | [![ci-test-img]][ci-test-url] |  [![coverage-img]][coverage-url] | [![contributors-img]][contributors-url] |


[license-img]:     https://img.shields.io/badge/License-MPL--2.0-blue
[license-url]:     https://github.com/JuliaSmoothOptimizers/Penelopt.jl/blob/main/LICENSE
[docs-stable-img]:  https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]:  https://jso.dev/Penelopt.jl/stable
[docs-dev-img]:     https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]:     https://jso.dev/Penelopt.jl/dev
[ci-test-img]:      https://github.com/JuliaSmoothOptimizers/Penelopt.jl/actions/workflows/Test.yml/badge.svg?branch=main
[ci-test-url]:      https://github.com/JuliaSmoothOptimizers/Penelopt.jl/actions/workflows/Test.yml?query=branch%3Amain
[coverage-img]:     https://codecov.io/gh/JuliaSmoothOptimizers/Penelopt.jl/branch/main/graph/badge.svg
[coverage-url]:     https://codecov.io/gh/JuliaSmoothOptimizers/Penelopt.jl
[contributors-img]: https://img.shields.io/github/all-contributors/JuliaSmoothOptimizers/Penelopt.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square
[contributors-url]: #contributors

## Content

Penelopt.jl is a software package for large-scale, equality-constrained, [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming).
It is designed to find (local) solutions of mathematical optimization problems of the form
```math
    \min f(x) \quad \text{s.t.} \ c(x) = 0.
```
where $f: R^n \to R$ and $c : R^n \to R^m$ are the objective and the constraint function, respectively.
Both $f$ and $c$ can be nonlinear and nonconvex but they are assumed to be continuously differentiable.
Moreover, it is assumed that $m \leq n$.

The solver is based on the exact [penalty algorithm](https://en.wikipedia.org/wiki/Penalty_method).

## How to Cite

If you use Penelopt.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/Penelopt.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://jso.dev/Penelopt.jl/dev/90-contributing/)

---