# [HSL tutorial](@id hsl-tutorial)

This tutorial shows how to use the [HSL](https://www.hsl.rl.ac.uk/) routine [MA57](https://www.hsl.rl.ac.uk/catalogue/ma57.html) as the linear solver of `Penelopt.jl`.

!!! note "Code not executed"
    The HSL libraries require a license, so the code on this page is not run when the documentation is built.

!!! warning "Extensions"
    `Penelopt.jl` uses an [extension](https://docs.julialang.org/en/v1/manual/code-loading/#man-extensions) to load MA57 through [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl). Therefore, you **need** to load [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl), together with a functional `HSL_jll`. Our algorithm will throw a warning and switch to the default [MUMPS](https://mumps-solver.org/index.php?page=doc) solver if you try to use MA57 without them.

## 1. Install HSL_jll and HSL.jl

`HSL_jll.jl` is not available from the General registry. Obtain it from [https://licences.stfc.ac.uk/product/libhsl](https://licences.stfc.ac.uk/product/libhsl) (free for academic use), unpack it, and add it to your environment together with HSL.jl:

```julia
using Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
Pkg.add("HSL")
```

See the [HSL.jl documentation](https://github.com/JuliaSmoothOptimizers/HSL.jl) for details.

## 2. Load HSL

```julia
using HSL_jll
using HSL

HSL.LIBHSL_isfunctional()  # should return true
```

## 3. Solve with Penelopt

Any `AbstractNLPModel` works; here we use a [CUTEst](CUTEst.md) problem.

```julia
using CUTEst, Penelopt

nlp = CUTEstModel("MSS1")
stats = L2Penalty(nlp; linear_solver = "ma57", print_level = 1)

println("status    : ", stats.status)
println("objective : ", stats.objective)

finalize(nlp)
```

!!! tip "How To Check?"
    You can verify that MA57 is correctly being used by inspecting the [output](../outputs.md) of the solver with the [option](../options.md) `print_level > 1`.
