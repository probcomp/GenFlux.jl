module Simple

using Gen
include("../src/GenFlux.jl")
using .GenFlux
using Flux

g = @genflux Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
tr = simulate(g, (rand(10), ))
display(tr)

end # module
