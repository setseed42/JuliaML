module DesicionTrees
using ..ImpurityCalculations, ..LeafValueCalculations, ..Structures, ..XGBlosses
export regression_tree, classification_tree, xgb_regression_tree, TreeSettings

struct SplitCriteria
    feature_i::Int64
    threshold::Float64
end

struct BestSets
    left::Union{Dataset,Missing}
    right::Union{Dataset,Missing}
end

struct DecisionNode
    criteria::Union{SplitCriteria,Missing}
    value::Union{Int64,Float64,Missing}
    true_branch::Union{DecisionNode,Missing}
    false_branch::Union{DecisionNode,Missing}
end


struct TreeSettings
    min_samples_split::Int64
    min_impurity::Float64
    max_depth::Int64
end

struct ImpurityInfo
    impurity::Float64
    criteria::SplitCriteria
    sets::BestSets
end

function divide_on_feature(data::Dataset,
    feature_i::Int64,
    threshold::Float64
    )::Tuple{Dataset, Dataset}
    split_func(sample) = sample[:,feature_i] .>= threshold
    splitter = split_func(data.x)
    x_1 = data.x[splitter,:]
    y_1 = data.y[splitter]
    x_2 = data.x[.!splitter,:]
    y_2 = data.y[.!splitter]
    Dataset(x_1, y_1), Dataset(x_2, y_2)
end

function find_larger_impurity(
    curr_impurity::ImpurityInfo,
    prev_impurity::ImpurityInfo
    )::ImpurityInfo

    if curr_impurity.impurity > prev_impurity.impurity
        return curr_impurity
    else
        return prev_impurity
    end
end

function desition_tree(
    impurity_calculation,
    leaf_value_calculation,
    tree_settings::TreeSettings
    )
    root = missing

    function build_tree(data::Dataset, current_depth::Int64=0)::DecisionNode
        largest_impurity = 0
        best_criteria = missing
        best_sets = missing
        n_samples, n_features = size(data.x)

        function calc_feature_impurity(feature_i::Int64)::ImpurityInfo
            feature_values = data.x[:,feature_i]
            unique_values = unique(feature_values)
            function calc_threshold_impurity(threshold::Float64)::ImpurityInfo
                left, right = divide_on_feature(data, feature_i, threshold)
                if (size(left.x, 1) > 0) & (size(right.x, 1) > 0)
                    impurity = impurity_calculation(data.y, left.y, right.y)
                    criteria = SplitCriteria(feature_i, threshold)
                    sets = BestSets(
                        left,
                        right
                    )
                else
                    impurity = 0.
                    criteria = SplitCriteria(feature_i, threshold)
                    sets = BestSets(missing, missing)
                end
                ImpurityInfo(impurity, criteria, sets)
            end
            mapreduce(
                calc_threshold_impurity,
                find_larger_impurity,
                unique_values
            )
        end
        if (n_samples >= tree_settings.min_samples_split) & (current_depth <= tree_settings.max_depth)
            best_impurity_info = mapreduce(
                calc_feature_impurity,
                find_larger_impurity,
                1:n_features
            )
            largest_impurity = best_impurity_info.impurity
        end

        if largest_impurity > tree_settings.min_impurity
            true_branch = build_tree(
                best_impurity_info.sets.left,
                current_depth + 1
            )
            false_branch = build_tree(
                best_impurity_info.sets.right,
                current_depth + 1
            )
            return DecisionNode(
                best_impurity_info.criteria,
                missing,
                true_branch,
                false_branch
            )
        end
        leaf_value = leaf_value_calculation(data.y)
        return DecisionNode(
            missing,
            leaf_value,
            missing,
            missing
        )
    end

    function predict_value(x::Array{Float64}, tree=missing)::Union{Float64, Int64}
        if ismissing(tree)
            tree = root
        end
        if !ismissing(tree.value)
            return tree.value
        end
        feature_value = x[tree.criteria.feature_i]
        branch = tree.false_branch
        feature_value_type = typeof(feature_value)

        if feature_value >= tree.criteria.threshold
            branch = tree.true_branch
        elseif feature_value == tree.criteria.threshold
            branch = tree.true_branch
        end
        predict_value(x, branch)
    end

    function fit(data::Dataset, loss=missing)
        root = build_tree(data)
        predict(x::Array{Float64}) = mapslices(predict_value, x, dims=2)[:]
        return predict
    end
    return fit
end


function regression_tree(tree_settings::TreeSettings)
    impurity_calculation = calculate_variance_reduction
    leaf_value_calculation = mean_of_y
    desition_tree(impurity_calculation, leaf_value_calculation, tree_settings)
end

function classification_tree(tree_settings::TreeSettings)
    impurity_calculation = calculate_information_gain
    leaf_value_calculation = majority_vote
    desition_tree(impurity_calculation, leaf_value_calculation, tree_settings)
end

function xgb_regression_tree(loss::XGBloss, tree_settings::TreeSettings)
    impurity_calculation = gain_by_taylor(loss)
    leaf_value_calculation = approximate_update(loss)
    desition_tree(impurity_calculation, leaf_value_calculation, tree_settings)
end


end
