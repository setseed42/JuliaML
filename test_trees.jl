include("RandomForest.jl")
include("Datasets.jl")
using .Datasets
using .RandomForest
using Plots
using Statistics
train_classif, test_classif = Datasets.make_classif_dataset()
train_reg, test_reg = Datasets.make_linreg_dataset(100, 5, 0.1)

n_estimators=1000
feature_sample=1.
min_samples_split=4
min_gain=10^-7
max_depth=typemax(Int64)
subsample_size=1.

random_forest_regressor = RandomForest.random_forest_regression(
    n_estimators,
    feature_sample,
    min_samples_split,
    min_gain,
    max_depth,
    subsample_size)

desicion_tree_regressor = RandomForest.desicion_tree_regression(
    min_samples_split,
    min_gain,
    max_depth)

function train_regressor(regressor, with_plots=false)
    trained_model = regressor(train_reg.x, train_reg.y)
    train_preds = trained_model(train_reg.x)
    test_preds = trained_model(test_reg.x)
    test_rmse = sqrt(mean((test_preds-test_reg.y) .^2))
    println("Test RMSE: $test_rmse")
    if with_plots
        scatter(test_preds, test_reg.y)
        line = minimum(hcat(test_preds, test_reg.y)):0.1:maximum(hcat(test_preds, test_reg.y))
        plot!(line, line)
    end
end


train_regressor(random_forest_regressor, true)
train_regressor(desicion_tree_regressor, true)



n_estimators=3
feature_sample=1.
min_samples_split=2
min_gain=10^-7
max_depth=typemax(Int64)
subsample_size=1.

random_forest_classifier = RandomForest.random_forest_classifiction(
    n_estimators,
    feature_sample,
    min_samples_split,
    min_gain,
    max_depth,
    subsample_size)

desicion_tree_classifier = RandomForest.desicion_tree_classification(
        min_samples_split,
        min_gain,
        max_depth)

function train_classifier(classifier, with_plots=false)
    trained_model = classifier(train_classif.x, train_classif.y)
    train_preds = trained_model(train_classif.x)
    test_preds = trained_model(test_classif.x)
    test_correctly_classified = test_preds .== test_classif.y
    test_accuracy = sum(test_correctly_classified) / length(test_correctly_classified)
    println("Test accuracy: $test_accuracy")
    if with_plots
        cats = length(unique(train_classif.y))
        histogram2d(test_classif.y, test_preds, nbins=cats)
    end
end


train_classifier(random_forest_classifier, true)
train_classifier(desicion_tree_classifier, true)
