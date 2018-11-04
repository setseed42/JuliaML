module DesitionTrees
include("ImpurityCalculations.jl")
include("LeafValueCalculations.jl")
include("Structures.jl")

using .ImpurityCalculations, .LeafValueCalculations, .Structures
export regression_tree, classification_tree

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
    value::Union{Array{Float64},Float64,Missing}
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

function divide_on_feature(x::Array{Float64},
    feature_i::Int64,
    threshold::Float64
    )::Tuple{Array{Float64}, Array{Float64}}

    split_func(sample) = sample[:,feature_i] .>= threshold
    x_1 = x[split_func(x),:]
    x_2 = x[.!split_func(x),:]
    x_1, x_2
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
    tree_settings = TreeSettings(2, 10^-7, typemax(Int64))::TreeSettings,
    loss=missing
    )
    root = missing

    function build_tree(data::Dataset, current_depth::Int64=0)::DecisionNode
        largest_impurity = 0
        best_criteria = missing
        best_sets = missing
        Xy = hcat(data.x, data.y)
        n_samples, n_features = size(data.x)

        function calc_feature_impurity(feature_i::Int64)::ImpurityInfo
            feature_values = data.x[:,feature_i]
            unique_values = unique(feature_values)
            function calc_threshold_impurity(threshold::Float64)::ImpurityInfo
                Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                if (size(Xy1, 1) > 0) & (size(Xy2, 1) > 0)
                    y1 = Xy1[:,end]
                    y2 = Xy2[:,end]
                    impurity = impurity_calculation(data.y, y1, y2)
                    criteria = SplitCriteria(feature_i, threshold)
                    sets = BestSets(
                        Dataset(Xy1[:,1:n_features], y1),
                        Dataset(Xy2[:,1:n_features], y2)
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

    function predict_value(x::Array{Float64}, tree=missing)
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

    function fit(x::Array{Float64}, y::Union{Array{Float64}, Array{Int64}}, loss=missing)
        data = Dataset(x, y)
        root = build_tree(data)
        predict(x::Array{Float64})::Array{Float64} = mapslices(predict_value, x, dims=2)
        return predict
    end
    return fit
end


function regression_tree(
    min_samples_split::Int64=2,
    min_impurity::Float64=10^-7,
    max_depth::Int64=typemax(Int64)
    )
    impurity_calculation = calculate_variance_reduction
    leaf_value_calculation = mean_of_y
    tree_settings = TreeSettings(
        min_samples_split,
        min_impurity,
        max_depth
    )
    desition_tree(impurity_calculation, leaf_value_calculation, tree_settings)
end

function classification_tree(
    min_samples_split::Int64=2,
    min_impurity::Float64=10^-7,
    max_depth::Int64=typemax(Int64)
    )
    impurity_calculation = calculate_information_gain
    leaf_value_calculation = majority_vote
    tree_settings = TreeSettings(
        min_samples_split,
        min_impurity,
        max_depth
    )
    desition_tree(impurity_calculation, leaf_value_calculation, tree_settings)
end

end
