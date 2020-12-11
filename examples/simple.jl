module Simple

using Gen
include("../src/GenFlux.jl")
using .GenFlux
using Flux

g = @genflux Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
tr = simulate(g, (rand(10), ))

# Grads.
ag_1 = Gen.accumulate_param_gradients!(tr, (1.0, 3.0), 1.0)
ag_2, _, _ = Gen.choice_gradients(tr, select(), (1.0, 3.0))
@assert ag_1 == ag_2

end # module
