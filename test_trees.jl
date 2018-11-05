include("RandomForest.jl")
include("Datasets.jl")
using .Datasets
using .RandomForest
using Plots

train_classif, test_classif = Datasets.make_classif_dataset()
train_reg, test_reg = Datasets.make_linreg_dataset(1000, 5, 0.1)

n_estimators=100
feature_sample=missing
min_samples_split=2
min_gain=10^-7
max_depth=5
subsample_size=0.7

model = RandomForest.random_forest_regression(
    n_estimators,
    feature_sample,
    min_samples_split,
    min_gain,
    max_depth,
    subsample_size)
trained_model = model(train_reg.x, train_reg.y)
predictions = trained_model(test_reg.x)
scatter(predictions, test_reg.y)

n_estimators=100
feature_sample=1.
min_samples_split=2
min_gain=10^-7
max_depth=typemax(Int64)
subsample_size=1.

model = RandomForest.random_forest_classifiction(
    n_estimators,
    feature_sample,
    min_samples_split,
    min_gain,
    max_depth,
    subsample_size)

trained_model = model(train_classif.x, train_classif.y)
cats = length(unique(train_classif.y))
train_preds = trained_model(train_classif.x)
histogram2d(train_classif.y, train_preds, nbins=cats)
test_preds = trained_model(test_classif.x)
histogram2d(test_classif.y, test_preds, nbins=cats)

model = RandomForest.desition_tree_classification(
    min_samples_split,
    min_gain,
    max_depth)

trained_model = model(train_classif.x, train_classif.y)
cats = length(unique(train_classif.y))
train_preds = trained_model(train_classif.x)
histogram2d(train_classif.y, train_preds, nbins=cats)
test_preds = trained_model(test_classif.x)
histogram2d(test_classif.y, test_preds, nbins=cats)
