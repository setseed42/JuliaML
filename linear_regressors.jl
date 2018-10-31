include("regularizers.jl")
include("optimizers.jl")
include("transformers.jl")
include("losses.jl")
include("splitters.jl")

module LinearRegressors
using Main.Regularizers
using Main.Optimizers
using Main.Transformers
using Main.Losses
using Main.Splitters

export linear_regression, lasso_regression, ridge_regression, elastic_net

function regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
    function trainer(x, y)
        x_train, y_train, x_test, y_test = train_test_split(x, y, 0.7)
        x_transformer = transformer(x_train)
        x_train = x_transformer(x_train)
        x_test = x_transformer(x_test)
        optimizer = gradient_descent(early_stopping, learning_rate)
        w = optimizer(x_train, y_train, x_test, y_test, regularizer, mse)
        predict(x) = x_transformer(x) * w
        return predict
    end
    return trainer
end


function linear_regression(early_stopping, learning_rate, batch_size)
    transformer = chain_transformers([
        standard_scaler(),
        add_bias()
    ])
    reg(w) = 0
    reg_gradient(w) = zeros(length(w))
    regularizer = (reg, reg_gradient)
    return regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
end

function lasso_regression(early_stopping, learning_rate, alpha, batch_size, degree)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = l1_regularization(alpha)
    return regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
end

function polynomial_regression(early_stopping, learning_rate, batch_size, degree)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    reg(w) = 0
    reg_gradient(w) = zeros(length(w))
    regularizer = (reg, reg_gradient)
    return regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
end


function ridge_regression(early_stopping, learning_rate, alpha, batch_size)
    transformer = chain_transformers([
        standard_scaler(),
        add_bias()
    ])
    regularizer = l2_regularization(alpha)
    return regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
end

function polynomial_ridge_regression(early_stopping, learning_rate, alpha, batch_size, degree)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = l2_regularization(alpha)
    return regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
end

function elastic_net(early_stopping, learning_rate, alpha, l1_ratio, batch_size, degree)
    transformer = chain_transformers([
        polynomial_features(degree),
        standard_scaler(),
        add_bias()
    ])
    regularizer = l1_l2_regularization(alpha, l1_ratio)
    return regressor(transformer, regularizer, early_stopping, learning_rate, batch_size)
end

end
