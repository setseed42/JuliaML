module Transformers
export add_bias, minmax_scaler, standard_scaler, chain_transformers, polynomial_features
using Statistics
using Combinatorics


function chain_transformers(transformers)
    function trainer(array::Array{Float64})
        trained_transformers = []
        for transformer in transformers
            trained_transformer = transformer(array)
            array = trained_transformer(array)
            push!(trained_transformers, trained_transformer)
        end
        function transform(x)
            for transformer = trained_transformers
                x = transformer(x)
            end
            x
        end
        transform
    end
    trainer
end


function add_bias()
    function trainer(array::Array{Float64})
        transform(x::Array{Float64})::Array{Float64} = hcat(x, ones(size(x, 1)))
        transform
    end
    trainer
end


function minmax_scaler(min::Float64, max::Float64)
    function trainer(array::Array{Float64})
        max_values = maximum(array, dims=1)
        min_values = minimum(array, dims=1)
        result_range = max - min
        scale_down(x::Array{Float64})::Array{Float64} = (x .- min_values) ./ (max_values .- min_values)
        scale_up(x::Array{Float64})::Array{Float64} = x .* result_range .+ min
        transform(x::Array{Float64})::Array{Float64} = scale_up(scale_down(x))
        transform
    end
    trainer
end


function standard_scaler()
    function trainer(array::Array{Float64})
        mean = Statistics.mean(array, dims=1)
        std = Statistics.std(array, dims=1)
        transform(x::Array{Float64})::Array{Float64} = (x .- mean) ./ std
        transform
    end
    trainer
end


function polynomial_features(degree::Int64)
    function trainer(array::Array{Float64})
        n_features = size(array, 2)
        polynomials = collect(multiexponents(n_features, degree))
        function transform(x::Array{Float64})::Array{Float64}
            init_mat = ones(size(x, 1), size(polynomials, 1))
            for (init_mat_ix, combination) in enumerate(polynomials)
                for (x_ix, exponent) in enumerate(combination)
                    if exponent == 0
                        continue
                    end
                    init_mat[:,init_mat_ix] = init_mat[:,init_mat_ix] .* x[:,x_ix] .^ exponent
                end
            end
            init_mat
        end
        transform
    end
    trainer
end


end
