module Datasets
export make_linreg_dataset
include("Transformers.jl")
include("Structures.jl")
include("Splitters.jl")
using Statistics
using Distributions
using Random
using RDatasets
using .Transformers
using .Structures
using .Splitters
function make_linreg_dataset(samples::Int64, features::Int64, noise::Float64)::Tuple{Dataset,Dataset}
    x = rand(Float64, (samples, features))
    xtra = hcat(x, ones(samples))
    w = rand(Float64, size(xtra, 2))
    noise = rand(Uniform(-noise, noise), samples)
    train, test = train_test_split(
        Dataset(x, xtra * w + noise),
        0.7,
    )
    return train, test
end

function make_classif_dataset()::Tuple{Dataset, Dataset}
    iris = dataset("datasets", "iris")
    x = convert(Array, iris[:,1:4])
    y = convert(Array, iris[:,end])
    y_transformer, _ = Transformers.label_encode()(y)
    y_transformed = y_transformer(y)
    train, test = train_test_split(
        Dataset(x, y_transformed),
        0.7,
    )
    return train, test
end


end
