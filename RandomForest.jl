module RandomForest
include("Structures.jl")
include("ImpurityCalculations.jl")
include("LeafValueCalculations.jl")
include("XGBlosses.jl")
include("DesitionTrees.jl")
include("Splitters.jl")

using Random, StatsBase, Statistics
using .DesitionTrees, .Structures, .Splitters, .ImpurityCalculations, .LeafValueCalculations, .XGBlosses
export random_forest_classifiction, random_forest_regression, desition_tree_classification


struct FittedEstimator
    fitted_estimator
    feature_indexes
end

function random_forest(
    predict_summary,
    estimator,
    n_estimators::Int64=100,
    feature_sample=missing::Union{Missing,Float64},
    subsample_size::Float64=0.5
    )

    function fit(x, y)
        data = Dataset(x, y)
        n_features = size(data.x, 2)

        if ismissing(feature_sample)
            max_features = floor(Int64, sqrt(n_features))
        else
            max_features = floor(Int64, n_features * feature_sample)
        end

        function fit_estimator()::FittedEstimator
            subset = get_random_subset(data, subsample_size)
            feature_indexes = shuffle(1:n_features)[1:max_features]
            bagged_x = subset.x[:,feature_indexes]
            fitted_estimator = estimator(Dataset(bagged_x, subset.y))
            FittedEstimator(fitted_estimator, feature_indexes)
        end

        fitted_estimators = map(x -> fit_estimator(), 1:n_estimators)
        function predict_estimator(x, fitted_estimator::FittedEstimator)
            bagged_x = x[:, fitted_estimator.feature_indexes]
            y_pred = fitted_estimator.fitted_estimator(bagged_x)
            y_pred
        end

        function predict(x)
            y_pred = zeros(size(x, 1), length(fitted_estimators))
            for (i, estimator) in enumerate(fitted_estimators)
                y_pred[:,i] = predict_estimator(x, estimator)
            end
            mapslices(predict_summary, y_pred, dims=2)[:]
        end
        predict
    end
    fit
end

function random_forest_classifiction(
    n_estimators::Int64=100,
    feature_sample=missing,
    min_samples_split::Int64=2,
    min_gain::Float64=10^-7,
    max_depth::Int64=typemax(Int64),
    subsample_size::Float64=0.5
    )
    tree_settings = TreeSettings(
        min_samples_split,
        min_gain,
        max_depth
    )
    estimator = classification_tree(tree_settings)
    random_forest(
        mode,
        estimator,
        n_estimators,
        feature_sample,
        subsample_size
    )
end

function random_forest_regression(
    n_estimators::Int64=100,
    feature_sample=missing,
    min_samples_split::Int64=2,
    min_gain::Float64=10^-7,
    max_depth::Int64=typemax(Int64),
    subsample_size::Float64=0.5
    )
    tree_settings = TreeSettings(
        min_samples_split,
        min_gain,
        max_depth
    )
    estimator = regression_tree(tree_settings)
    random_forest(
        mean,
        estimator,
        n_estimators,
        feature_sample,
        subsample_size
    )
end

function desition_tree_classification(
    min_samples_split::Int64=2,
    min_impurity::Float64=10^-7,
    max_depth::Int64=typemax(Int64),
    )
    tree_settings = TreeSettings(
        min_samples_split,
        min_impurity,
        max_depth
    )
    model = classification_tree(tree_settings)
    function train(x, y)
        data = Dataset(x, y)
        model(data)
    end
    train
end
end
