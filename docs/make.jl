using Penelopt
using Documenter

DocMeta.setdocmeta!(Penelopt, :DocTestSetup, :(using Penelopt); recursive = true)

makedocs(;
  modules = [Penelopt],
  authors = "Maxence Gollier maxence-2.gollier@polymtl.ca",
  repo = "https://github.com/JuliaSmoothOptimizers/Penelopt.jl/blob/{commit}{path}#{line}",
  sitename = "Penelopt.jl",
  format = Documenter.HTML(;
    canonical = "https://jso.dev/Penelopt.jl",
    assets = ["assets/link-icons.css"],
    collapselevel = 1,
  ),
  workdir = joinpath(@__DIR__, "src"),
  pages = [
    "Home" => "index.md",
    "Options" => "options.md",
    "Outputs" => "outputs.md",
    "Performance" => "performance.md",
    "Callbacks" => "callbacks.md",
    "Tutorials" => [
      "AMPL" => "tutorials/AMPL.md",
      "CUTEst" => "tutorials/CUTEst.md",
      "JuMP" => "tutorials/JuMP.md",
      "HSL" => "tutorials/HSL.md",
      "LDLFactorizations" => "tutorials/LDLFactorizations.md",
    ],
    "Developers" =>
      ["Contributing" => "90-contributing.md", "Developing" => "91-developer.md"],
    "Reference" => "95-reference.md",
  ],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/Penelopt.jl", push_preview = true)
