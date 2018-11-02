module LinearRegressors
include("Structures.jl")
include("Regularizers.jl")
include("Optimizers.jl")
include("Transformers.jl")
include("Losses.jl")
include("Splitters.jl")

using .Regularizers, .Optimizers, .Transformers, .Losses, .Splitters, .Structures
export linear_regression, lasso_regression, ridge_regression, elastic_net, polynomial_regression, polynomial_ridge_regression


function regressor(transformer, regularizer::Regularizer, early_stopping::Int64, learning_rate::Float64)
    function trainer(x::Array{Float64}, y::Array{Float64})
        data = Dataset(x, y)
        train, test = train_test_split(data, 0.7)
        x_transformer = transformer(train.x)
        x_train = x_transformer(train.x)
        x_test = x_transformer(test.x)
        optimizer = gradient_descent(early_stopping, learning_rate)
        train = Dataset(x_train, train.y)
        test = Dataset(x_test, test.y)
        w = optimizer(train, test, regularizer, mse)
        predict(x) = x_transformer(x) * w
        predict
    end
    trainer
end


function linear_regression(early_stopping::Int64, learning_rate::Float64)
    transformer = chain_transformers([
        standard_scaler(),
        add_bias()
    ])
    regularizer = Regularizer(
        w -> 0,
        w -> zeros(length(w))
    )
    regressor(transformer, regularizer, early_stopping, learning_rate)
end

function lasso_regression(early_stopping::Int64, learning_rate::Float64, alpha::Float64, degree::Int64)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = l1_regularization(alpha)
    regressor(transformer, regularizer, early_stopping, learning_rate)
end

function polynomial_regression(early_stopping::Int64, learning_rate::Float64, degree::Int64)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = Regularizer(
        w -> 0,
        w -> zeros(length(w))
    )
    regressor(transformer, regularizer, early_stopping, learning_rate)
end


function ridge_regression(early_stopping::Int64, learning_rate::Float64, alpha::Float64)
    transformer = chain_transformers([
        standard_scaler(),
        add_bias()
    ])
    regularizer = l2_regularization(alpha)
    regressor(transformer, regularizer, early_stopping, learning_rate)
end

function polynomial_ridge_regression(early_stopping::Int64, learning_rate::Float64, alpha::Float64, degree::Int64)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = l2_regularization(alpha)
    regressor(transformer, regularizer, early_stopping, learning_rate)
end

function elastic_net(early_stopping::Int64, learning_rate::Float64, alpha::Float64, l1_ratio::Float64, degree::Int64)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = l1_l2_regularization(alpha, l1_ratio)
    regressor(transformer, regularizer, early_stopping, learning_rate)
end

end
