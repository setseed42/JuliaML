module Transformers
export add_bias, minmax_scaler, standard_scaler, chain_transformers, polynomial_features
using Statistics
using Combinatorics


function chain_transformers(transformers)
    function trainer(array)
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
            return x
        end
        return transform
    end
    return trainer
end


function add_bias()
    function trainer(array)
        transform(x) = hcat(x, ones(size(x, 1)))
        return transform
    end
    return trainer
end


function minmax_scaler(min, max)
    function trainer(array)
        max_values = maximum(array, dims=1)
        min_values = minimum(array, dims=1)
        result_range = max - min
        scale_down(x) = (x .- min_values) ./ (max_values .- min_values)
        scale_up(x) = x .* result_range .+ min
        transform(x) = scale_up(scale_down(x))
        return transform
    end
    return trainer
end


function standard_scaler()
    function trainer(array)
        mean = Statistics.mean(array, dims=1)
        std = Statistics.std(array, dims=1)
        transform(x) = (x .- mean) ./ std
        return transform
    end
    return trainer
end


function polynomial_features(degree)
    function trainer(array)
        n_features = size(array, 2)
        polynomials = collect(multiexponents(n_features, degree))
        function transform(x)
            init_mat = ones(size(x, 1), size(polynomials, 1))
            for (init_mat_ix, combination) in enumerate(polynomials)
                for (x_ix, exponent) in enumerate(combination)
                    if exponent == 0
                        continue
                    end
                    init_mat[:,init_mat_ix] = init_mat[:,init_mat_ix] .* x[:,x_ix] .^ exponent
                end
            end
            return init_mat
        end
        return transform
    end
    return trainer
end


end
