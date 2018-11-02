include("LinearRegressors.jl")
using .LinearRegressors
using Statistics
using Plots
using Distributions
using Random

function make_dataset(samples::Int64, features::Int64, noise::Float64)::Tuple{Array{Float64},Array{Float64, 1}}
    x = rand(Float64, (samples, features))
    xtra = hcat(x, ones(samples))
    w = rand(Float64, size(xtra, 2))
    noise = rand(Uniform(-noise, noise), samples)
    return x, xtra * w + noise
end

x, y = make_dataset(10000, 100, 0.1)
println(typeof(make_dataset))

# Settings
early_stopping = 1000
learning_rate = 10 ^ -4
alpha = 0.1
l1_ratio = 0.1
degree = 1

println("Testing linear regression")
model = LinearRegressors.linear_regression(early_stopping, learning_rate)
trained_model = model(x, y)
predictions = trained_model(x)

println("Testing lasso regression")
model = lasso_regression(early_stopping, learning_rate, alpha, degree)
trained_model = model(x, y)
predictions = trained_model(x)

println("Testing polynomial regression")
model = polynomial_regression(early_stopping, learning_rate, degree)
trained_model = model(x, y)
predictions = trained_model(x)

println("Testing ridge regression")
model = ridge_regression(early_stopping, learning_rate, alpha)
trained_model = model(x, y)
predictions = trained_model(x)

println("Testing polynomial ridge regression")
model = polynomial_ridge_regression(early_stopping, learning_rate, alpha, degree)
trained_model = model(x, y)
predictions = trained_model(x)

println("Testing elastic_net")
model = elastic_net(early_stopping, learning_rate, alpha, l1_ratio, degree)
trained_model = model(x, y)
predictions = trained_model(x)
