module MNIST

# ------------ load mnist ------------ #

import Random
Random.seed!(1)

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata()

mutable struct DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, Random.shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float64, batch_size, 1, 28, 28)
    y = Vector{Int}(undef, batch_size)
    for i=1:batch_size
        x[i, 1, :, :] = train_x[:,:,loader.cur_id]
        y[i] = train_y[loader.cur_id] + 1
        loader.cur_id = (loader.cur_id % 60000) + 1
    end
    x, y
end

function load_test_set()
    test_x, test_y = MLDatasets.MNIST.testdata()
    N = length(test_y)
    x = zeros(Float64, N, 1, 28, 28)
    y = Vector{Int}(undef, N)
    for i=1:N
        x[i, 1, :, :] = test_x[:,:,i]
        y[i] = test_y[i]+1
    end
    x, y
end

const loader = DataLoader()

(test_x, test_y) = load_test_set()

# ------------ Model ------------ #

using Gen
include("../src/GenFlux.jl")
using .GenFlux
using Flux

g = @genflux Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
tr = simulate(g, (rand(10), ))
display(tr)

end # module
