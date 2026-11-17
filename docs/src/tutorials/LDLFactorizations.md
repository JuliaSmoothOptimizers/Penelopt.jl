# [LDLFactorizations tutorial](@id ldlt-tutorial)

This tutorial shows how to use the [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) linear solver to solve a problem from the
[CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) (see [this](CUTEst.md) tutorial) collection with `Penelopt.jl`.

!!! warning "Extensions"
    `Penelopt.jl` uses an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions) to load the [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) linear solver. Therefore, you **need** to load [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl). Our algorithm will throw a warning and switch to the default [MUMPS](https://mumps-solver.org/index.php?page=doc) solver if you try to use LDLFactorizations without loading the required package.

## 1. Load a CUTEst problem

In this example, we choose a medium size problem **MSS1**.

```@example ldlt
using CUTEst

nlp = CUTEstModel("MSS1")

nothing # hide
```

## 2. Load LDLFactorizations

```@example ldlt
using LDLFactorizations
```

## 3. Solve with Penelopt

```@example ldlt
using Penelopt

stats = L2Penalty(nlp; linear_solver = :ldlt, print_level = 1)

nothing # hide
```

```@example ldlt
println("status    : ", stats.status)
println("objective : ", stats.objective)
println("solution  : ", stats.solution)

nothing # hide
```

## 4. Finalize the CUTEst model

Once the CUTEst problem has been used, you should finalize it, see the CUTEst [documentation](https://jso.dev/CUTEst.jl/stable/).

```@example ldlt
finalize(nlp)

nothing # hide
```

!!! tip "How To Check?"
    You can verify that LDLFactorizations.jl is correctly being used by inspecting the [output](../outputs.md) of the solver with the [option](../options.md) `print_level > 1`.
