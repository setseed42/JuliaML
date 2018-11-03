include("LinearRegressors.jl")
include("Datasets.jl")
using .LinearRegressors
using .Datasets
using Plots

test, train = Datasets.make_linreg_dataset(100, 10, 0.1)

# Settings
early_stopping = 1000
learning_rate = 10 ^ -4
reg_alpha = 0.1
l1_ratio = 0.1
degree = 1

println("Testing linear regression")
model = LinearRegressors.linear_regression(early_stopping, learning_rate)
trained_model = model(train.x, train.y)
predictions = trained_model(test.x)
lin_reg = scatter(predictions, test.y, title="lin_reg")

println("Testing lasso regression")
model = lasso_regression(early_stopping, learning_rate, reg_alpha, degree)
trained_model = model(train.x, train.y)
predictions = trained_model(test.x)
lasso_reg = scatter(predictions, test.y, title="lasso_reg")

println("Testing polynomial regression")
model = polynomial_regression(early_stopping, learning_rate, degree)
trained_model = model(train.x, train.y)
predictions = trained_model(test.x)
poly_reg = scatter(predictions, test.y, title="poly_reg")

println("Testing ridge regression")
model = ridge_regression(early_stopping, learning_rate, reg_alpha)
trained_model = model(train.x, train.y)
predictions = trained_model(test.x)
ridge_reg = scatter(predictions, test.y, title="ridge_reg")

println("Testing polynomial ridge regression")
model = polynomial_ridge_regression(early_stopping, learning_rate, reg_alpha, degree)
trained_model = model(train.x, train.y)
predictions = trained_model(test.x)
polyridge_reg = scatter(predictions, test.y, title="polyridge_reg")

println("Testing elastic_net")
model = elastic_net(early_stopping, learning_rate, reg_alpha, l1_ratio, degree)
trained_model = model(train.x, train.y)
predictions = trained_model(test.x)
elastic_reg = scatter(predictions, test.y, title="elastic_net")

plot(lin_reg,
    lasso_reg,
    poly_reg,
    ridge_reg,
    polyridge_reg,
    elastic_reg,
    layout=(2, 3),
    label="")
