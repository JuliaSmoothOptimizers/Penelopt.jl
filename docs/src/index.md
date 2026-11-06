# Welcome to Penelopt.jl

```@meta
CurrentModule = Penelopt
```

```@docs
Penelopt.Penelopt
```

Penelopt.jl is a software package for large-scale, equality-constrained, [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming).
It is designed to find (local) solutions of mathematical optimization problems of the form
```math
    \underset{x \in \mathbb{R}^n}{\textup{minimize}} \ f(x) \quad \textup{subject to} \ c(x) = 0,
```
where $f: \mathbb{R}^n \to \mathbb{R}$ and $c : \mathbb{R}^n \to \mathbb{R}^m$ are the objective and the constraint function, respectively.
Both $f$ and $c$ can be nonlinear and nonconvex but they are assumed to be continuously differentiable.
Moreover, it is assumed that $m \leq n$.

Our solver is based on the exact [penalty method](https://en.wikipedia.org/wiki/Penalty_method).
It consists in transforming the equality-constrained problem above into a sequence of unconstrained penalized problems
```math
    \underset{x \in \mathbb{R}^n}{\textup{minimize}} \ f(x) + \tau_k g(c(x)).
```
where $\tau_k > 0$ is the penalty parameter and $g$ is the penalty function.
In this package, we use $g = \|\cdot\|_2$.

## Main Features

On a practical level, the package uses the standardized [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API to represent the non-linear programming (NLP) problem. If you want to figure out how to interface your NLP problem with [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl), we refer to their [documentation](https://jso.dev/NLPModels.jl/stable/).

!!! note "Using Other Modelling Languages"
    If your NLP is represented in one of the following formats/sets, no worries, you can easily use our solver!
    * [AMPL](https://ampl.com/wp-content/uploads/BOOK.pdf): Use [AmplNLReader.jl](https://github.com/JuliaSmoothOptimizers/AmplNLReader.jl). You should read [this tutorial](tutorials/AMPL.md).
    * [JuMP](https://jump.dev/JuMP.jl/stable/): Use [NLPModelsJuMP.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl). You should read [this tutorial](tutorials/JuMP.md).
    * [CUTEst](https://link.springer.com/article/10.1007/s10589-014-9687-3): Use [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl). You should read [this tutorial](tutorials/CUTEst.md).

Once the problem has been modeled, you can simply install Penelopt.jl and solve your problem! 🚀
```{julia}
# Interface your problem with the NLPModels.jl API
julia> nlp = ...

# Install
julia> using Pkg
julia> Pkg.add("Penelopt")
julia> using Penelopt

# Solve
julia> L2Penalty(nlp)
```
The example above uses the solver with its default settings.
Penelopt.jl also provides a variety of [options](options.md) for customizing the optimization process, and returns a [result object](outputs.md) containing the computed solution together with diagnostic information. 📖

For more advanced use cases, you can monitor or customize the optimization process through [callbacks](callbacks.md). 🔧

Finally, if performance is critical, you can find tips in the [performance section](performance.md). ⚡

!!! note "Choosing A Linear Solver Library"
    Our algorithm solves multiple [linear systems](https://en.wikipedia.org/wiki/System_of_linear_equations).
    By default, the linear system solver we use is [MUMPS](https://mumps-solver.org/index.php?page=doc).
    Our solver is interfaced with the following solvers too:
    * [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl): you can read [this tutorial](tutorials/LDLFactorizations.md) to use it.
    * [HSL-MA57](https://www.hsl.rl.ac.uk/catalogue/ma57.html).

## Credit

This documentation is largely inspired by the excellent documentation of [Ipopt](https://coin-or.github.io/Ipopt/), [Krylov.jl](https://jso.dev/Krylov.jl/stable/), and [Manopt.jl](https://manoptjl.org/stable/).
We gratefully acknowledge the work of the contributors to these projects, whose efforts have provided valuable examples of clear and comprehensive software documentation.

For anyone looking for inspiration when writing documentation for their own packages, we highly recommend referring to these projects as excellent examples.
