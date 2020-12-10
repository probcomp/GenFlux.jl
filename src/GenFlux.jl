module GenFlux

using Gen
using Flux

const Model = Union{Chain, Dense, RNN, LSTM, GRU, Conv}

# ------------ Trace ------------ #

struct FluxTrace{R} <: Gen.Trace
    gen_fn::GenerativeFunction
    args::Tuple
    retval::R
end

@inline Gen.get_args(trace::FluxTrace) = trace.args
@inline Gen.get_retval(trace::FluxTrace) = trace.retval
@inline Gen.get_score(trace::FluxTrace) = 0.0
@inline Gen.get_choices(trace::FluxTrace) = EmptyChoiceMap()
@inline Gen.get_gen_fn(trace::FluxTrace) = trace.gen_fn

# ------------ Generative function ------------ #

struct FluxGenerativeFunction{R} <: Gen.GenerativeFunction{R, FluxTrace{R}}
    model::Model
    params
    FluxGenerativeFunction{R}(model) = new{R}(model, device, Flux.params(model))
    FluxGenerativeFunction(model) = new{Any}(model, device, Flux.params(model))
end

# ------------ GFI ------------ #

function Gen.simulate(gen_fn::FluxGenerativeFunction, args::Tuple)
    ret = gen_fn.model(args)
    FluxTrace{typeof(ret)}(gen_fn, args, ret)
end

function Gen.generate(gen_fn::FluxGenerativeFunction, args::Tuple, ::ChoiceMap)
    trace = simulate(gen_fn, args)
    (trace, 0.0)
end

function Gen.propose(gen_fn::FluxGenerativeFunction, args::Tuple)
    trace = simulate(gen_fn, args)
    retval = get_retval(trace)
    (EmptyChoiceMap(), 0.0, retval)
end

@inline Gen.project(::FluxTrace, ::Selection) = 0.0

function Gen.update(trace::FluxTrace, args::Tuple, argdiffs::Tuple, ::ChoiceMap)
    if all(x -> x isa NoChange, argdiffs)
        return (trace, 0., NoChange(), EmptyChoiceMap())
    end
    trace = simulate(trace.gen_fn, args)
    (trace, 0., UnknownChange(), EmptyChoiceMap())
end

function Gen.regenerate(trace::FluxTrace, args::Tuple, argdiffs::Tuple, ::Selection)
    if all(x -> x isa NoChange, argdiffs)
        return (trace, 0., NoChange())
    end
    trace = simulate(trace.gen_fn, args)
    (trace, 0., UnknownChange())
end

end # module
