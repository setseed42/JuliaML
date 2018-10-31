include("linear_regressors.jl")
using Statistics
using Plots
using Distributions

function make_dataset(samples, features, noise)
    x = rand(Float32, (samples, features))
    xtra = hcat(x, ones(samples))
    w = rand(Float32, size(xtra, 2))
    noise = rand(Uniform(-noise, noise), samples)
    return x, xtra * w + noise
end

x, y = make_dataset(1000, 10, 0.1)

# Test linear regression

early_stopping = 1000
learning_rate = 10 ^ -4
batch_size = 1

model = LinearRegressors.linear_regression(early_stopping, learning_rate, batch_size)
trained_model = model(x, y)
predictions = trained_model(x)
plotly() # Choose the Plotly.jl backend for web interactivity
plot(y, predictions, seriestype=:scatter, )
gui()

# Test elastic net

early_stopping = 1000
learning_rate = 10 ^ -4
alpha = 0
l1_ratio = 1
batch_size = 32
degree = 1

model = LinearRegressors.elastic_net(early_stopping, learning_rate, alpha, l1_ratio, batch_size, degree)
trained_model = model(x, y)
predictions = trained_model(x)
plotly() # Choose the Plotly.jl backend for web interactivity
plot(y, predictions, seriestype=:scatter, )
gui()
