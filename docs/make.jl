using Documenter, GenFlux

makedocs(modules=[GenFlux],
         sitename="GenFlux",
         authors="McCoy R. Becker and other contributors",
         pages=["API Documentation" => "index.md"])
deploydocs(repo = "github.com/probcomp/GenFlux.jl.git")
