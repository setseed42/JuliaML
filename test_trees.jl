include("DesitionTrees.jl")
include("Datasets.jl")
using .DesitionTrees
using .Datasets
using Plots

train_classif, test_classif = Datasets.make_classif_dataset()
train_reg, test_reg = Datasets.make_linreg_dataset(1000, 10, 0.1)


model = DesitionTrees.regression_tree(3, 10^-7, 40)
trained_model = model(train_reg.x, train_reg.y)
predictions = trained_model(test_reg.x)
scatter(predictions, test_reg.y)


model = DesitionTrees.classification_tree(2, 10^-7, typemax(Int64))
trained_model = model(train_classif.x, train_classif.y)
predictions = trained_model(test_classif.x)
scatter(predictions, test_classif.y)
