module Xgboost

include("Structures.jl")
include("ImpurityCalculations.jl")
include("LeafValueCalculations.jl")
include("XGBlosses.jl")
include("DesitionTrees.jl")

using .DesitionTrees, .XGBlosses
export regression_tree, classification_tree, xgboost, TreeSettings

function xgboost(
    tree_settings::TreeSettings,
    loss::XGBloss,
    n_estimators::Int64=200,
    learning_rate::Float64=0.001,
    )
    trees = []
    for i in 1:n_estimators
        tree = xgb_regression_tree(
            loss,
            tree_settings
        )
        push!(trees, tree)
    end
    function train(x, y)
        y_pred = zeros(size(y))
        trained_trees = []
        for tree in trees
            y_and_pred = hcat(y, y_pred)
            trained_tree = tree(x, y_and_pred)
            push!(trained_trees, trained_tree)
            update_pred = trained_tree(x)
            y_pred -= learning_rate .* update_pred
        end
        function predict(x)
            y_pred = missing
            for trained_tree in trained_trees
                update_pred = trained_tree(x)
                if ismissing(y_pred)
                    y_pred = np.zeros(size(update_pred))
                end
                y_pred -= learning_rate .* update_pred
            end
            y_pred
        end
        predict
    end
    train
end

function xgb_classifier(
    n_estimators::Int64=200,
    learning_rate::Float64=0.001,
    min_samples_split::Int64=2,
    min_impurity::Float64=1e-7,
    max_depth::Int64=2
    )
    loss = logisticloss()
    tree_settings = TreeSettings(
        min_samples_split,
        min_impurity,
        max_depth
    )
    xgboost(tree_settings, loss, n_estimators, learning_rate)
end

end
