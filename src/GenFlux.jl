module GenFlux

using Gen
using Flux
using Zygote

const FluxModel = Union{Chain, Dense, 
                        Flux.Recur, Flux.RNNCell, Flux.LSTMCell, Flux.GRUCell, 
                        Conv, ConvTranspose, DepthwiseConv, CrossCor, AdaptiveMaxPool, AdaptiveMeanPool, GlobalMaxPool, GlobalMeanPool, MaxPool, MeanPool}

const FluxOptimizer = Union{Descent, Momentum, Nesterov, RMSProp, Flux.ADAM, RADAM, AdaMax, OADAM, ADAGrad, ADADelta, AMSGrad, NADAM, AdaBelief, Flux.Optimiser}

# ------------ Trace ------------ #

# Possibly type for CuArrays.
struct FluxTrace{A} <: Gen.Trace
    gen_fn::GenerativeFunction
    args::Tuple
    retval::A
end

@inline Gen.get_args(trace::FluxTrace) = trace.args
@inline Gen.get_retval(trace::FluxTrace) = trace.retval
@inline Gen.get_score(trace::FluxTrace) = 0.0
@inline Gen.get_choices(trace::FluxTrace) = EmptyChoiceMap()
@inline Gen.get_gen_fn(trace::FluxTrace) = trace.gen_fn

# ------------ Generative function ------------ #

mutable struct FluxGenerativeFunction <: Gen.GenerativeFunction{Any, FluxTrace}
    model::FluxModel
    params::Vector
    params_grads::Vector
    restructure::Function # reset model after applying params_grads to params
    function FluxGenerativeFunction(model)
        ps, re = Flux.destructure(model)
        new(model, ps, Zygote.zero(ps), re)
    end
end

@inline (g::FluxGenerativeFunction)(args...) = Gen.get_retval(Gen.simulate(g, args))
Zygote.@adjoint function (g::FluxGenerativeFunction)(args...)
    ret = g(args...)
    back = ret_grad -> begin
        _, back = Zygote.pullback((m, x) -> m(x...), g.model, args)
        params_grads, arg_grads = back(ret_grad)
        (nothing, Flux.destructure(params_grads)[1], arg_grads...)
    end
    ret, back
end

@inline function accumulate!(g::FluxGenerativeFunction, scaler::Float64, v::Array)
    g.params_grads = g.params_grads + scaler * v
end

@inline Gen.accepts_output_grad(g::FluxGenerativeFunction) = true
@inline Gen.has_argument_grads(g::FluxGenerativeFunction) = (true, )
@inline Gen.get_params(g::FluxGenerativeFunction) = g.params

# ------------ GFI ------------ #

function Gen.simulate(gen_fn::FluxGenerativeFunction, args::Tuple)
    ret = gen_fn.model(args...)
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

# ------------ Gradients ------------ #

function backwards(trace::FluxTrace, retval_grad)
    model, args = get_gen_fn(trace), get_args(trace)
    ret, back = Zygote.pullback(model, args...)
    ps_grads, arg_grads = back(retval_grad)
    ps_grads, arg_grads
end

function Gen.choice_gradients(trace::FluxTrace, ::Selection, retval_grad)
    _, arg_grads = backwards(trace, retval_grad)
    (arg_grads, ), EmptyChoiceMap(), EmptyChoiceMap()
end

function Gen.accumulate_param_gradients!(trace::FluxTrace, retval_grad, multiplier = 1.0)
    params_grads, arg_grads = backwards(trace, retval_grad)
    g = get_gen_fn(trace)
    accumulate!(g, multiplier, params_grads)
    (arg_grads, )
end

# ------------ Learning ------------ #

struct FluxOptimizerState
    opt
    g
end

function Gen.init_update_state(conf::FixedStepGradientDescent, g::FluxGenerativeFunction, ::Any)
    opt = Flux.SGD(η = conf.learning_rate)
    FluxOptimizerState(opt, g)
end

function Gen.init_update_state(conf::Gen.ADAM, g::FluxGenerativeFunction, ::Any)
    opt = Flux.ADAM(conf.learning_rate, (conf.beta1, conf.beta2))
    FluxOptimizerState(opt, g)
end

@inline Gen.init_update_state(opt::O, g::FluxGenerativeFunction, ::Any) where O <: FluxOptimizer = FluxOptimizerState(opt, g)

function Gen.apply_update!(state::FluxOptimizerState)
    Flux.update!(state.opt, state.g.params, state.g.params_grads)
    state.g.model = state.g.restructure(state.g.params)
    state.g.params_grads = Zygote.zero(state.g.params_grads)
end

# ------------ Macro ------------ #

_genflux(expr) = Expr(:call, GlobalRef(GenFlux, :FluxGenerativeFunction), expr)
macro genflux(expr)
    new = _genflux(expr)
    esc(new)
end
export @genflux

end # module
