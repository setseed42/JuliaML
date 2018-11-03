module Transformers
export add_bias, minmax_scaler, standard_scaler, chain_transformers, polynomial_features
using Statistics
using Combinatorics


function chain_transformers(transformers)
    function trainer(array::Array{Float64})
        trained_transformers = []
        for transformer in transformers
            trained_transformer, _ = transformer(array)
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
        inverse_transform(x_transformed::Array{Float64})::Array{Float64} = x_transformed[:,1:end-1]
        transform, inverse_transform
    end
    trainer
end


function minmax_scaler(min::Float64, max::Float64)
    function trainer(array::Array{Float64})
        max_values = maximum(array, dims=1)
        min_values = minimum(array, dims=1)

        function scale_down(x::Array{Float64}, min, max)::Array{Float64}
            (x .- min) ./ (max .- min)
        end
        function scale_up(x::Array{Float64}, min, max)::Array{Float64}
            x .* (max .- min) .+ min
        end
        function transform(x::Array{Float64})::Array{Float64}
            scale_up(scale_down(x, min_values, max_values), min, max)
        end
        function inverse_transform(x::Array{Float64})::Array{Float64}
            scale_down(scale_up(x, min_values, max_values), min, max)
        end
        transform, inverse_transform
    end
    trainer
end


function standard_scaler()
    function trainer(array::Array{Float64})
        mean = Statistics.mean(array, dims=1)
        std = Statistics.std(array, dims=1)
        transform(x::Array{Float64})::Array{Float64} = (x .- mean) ./ std
        inverse_transform(x::Array{Float64})::Array{Float64} = x .* std .+ mean
        transform, inverse_transform
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
        #TODO
        inverse_transform(x) = x
        transform, inverse_transform
    end
    trainer
end

function label_encode()
    function trainer(array::Array{String})
        unique_labels = unique(array)

        label_map = Dict{String,Int64}()
        for (i, key) in enumerate(unique_labels)
            label_map[key] = i
        end

        inverse_map = Dict{Int64,String}()
        for (key, value) in label_map
            inverse_map[value] = key
        end
        get_value(key::String)::Int64 = label_map[key]
        get_key(value::Int64)::String = inverse_map[value]
        transform(label::Array{String})::Array{Int64} = map(get_value, label)
        inverse_transform(value::Array{Int64})::Array{String} = map(get_key, value)
        transform, inverse_transform
    end
    trainer
end


end
