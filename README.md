# GenFlux.jl

![Build Status](https://github.com/probcomp/GenFlux.jl/actions/workflows/CI.yml/badge.svg)
[![Link to Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://probcomp.github.io/GenFlux.jl/dev)

`GenFlux.jl` is Gen DSL which implements [the generative function interface](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1) to allow the usage of [Flux.jl models](https://github.com/FluxML/Flux.jl) as Gen generative functions.

---

([full example available here](https://github.com/femtomc/GenFlux.jl/blob/master/examples/mnist.jl))

```julia
g = @genflux Chain(Conv((5, 5), 1 => 10; init = glorot_uniform64),
                   MaxPool((2, 2)),
                   x -> relu.(x),
                   Conv((5, 5), 10 => 20; init = glorot_uniform64),
                   x -> relu.(x),
                   MaxPool((2, 2)),
                   x -> flatten(x),
                   Dense(320, 50; initW = glorot_uniform64),
                   Dense(50, 10; initW = glorot_uniform64),
                   softmax)
```

Now you can use `g` as a modelling component in your probabilistic programs:

```julia
@gen function f(xs::Vector{Float64})
    probs ~ g(xs)
    [{:y => i} ~ categorical(p |> collect) for (i, p) in enumerate(eachcol(probs))]
end
```

Allowing you to train the parameters of `g` via gradient descent [on the objective](https://www.gen.dev/dev/ref/gfi/#Gen.accumulate_param_gradients!):

```julia
update = ParamUpdate(Flux.ADAM(5e-5, (0.9, 0.999)), g)
for i = 1 : 1500
    # Create trace from data
    (xs, ys) = next_batch(loader, 100)
    constraints = choicemap([(:y => i) => y for (i, y) in enumerate(ys)]...)
    (trace, weight) = generate(f, (xs,), constraints)

    # Increment gradient accumulators
    accumulate_param_gradients!(trace)

    # Perform ADAM update and then resets gradient accumulators
    apply!(update)
    println("i: $i, weight: $weight")
end
```

```julia
test_accuracy = mean(f(test_x) .== test_y)
println("Test set accuracy: $test_accuracy")
# Test set accuracy: 0.9392
```
