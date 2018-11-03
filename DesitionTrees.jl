module DesitionTrees
include("ImpurityCalculations.jl")
include("LeafValueCalculations.jl")
include("Structures.jl")

using .ImpurityCalculations, .LeafValueCalculations, .Structures
export regression_tree, classification_tree

struct BestCriteria
    feature_i::Int64
    threshold::Float64
end

struct BestSets
    leftX::Array{Float64}
    lefty::Array{Float64}
    rightX::Array{Float64}
    righty::Array{Float64}
end

struct DecisionNode
    feature_i
    threshold
    value
    true_branch
    false_branch
end


struct TreeSettings
    min_samples_split::Int64
    min_impurity::Float64
    max_depth::Int64
end


function divide_on_feature(x::Array{Float64}, feature_i::Int64, threshold::Float64)::Tuple{Array{Float64}, Array{Float64}}
    split_func(sample) = sample[:,feature_i] .>= threshold
    x_1 = x[split_func(x),:]
    x_2 = x[.!split_func(x),:]
    x_1, x_2
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
        if (n_samples >= tree_settings.min_samples_split) & (current_depth <= tree_settings.max_depth)
            for feature_i = 1:n_features
                feature_values = data.x[:,feature_i]
                unique_values = unique(feature_values)
                for threshold in unique_values
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    if (size(Xy1, 1) > 0) & (size(Xy2, 1) > 0)
                        y1 = Xy1[:,end]
                        y2 = Xy2[:,end]
                        impurity = impurity_calculation(data.y, y1, y2)
                        if impurity > largest_impurity
                            largest_impurity = impurity
                            best_criteria = BestCriteria(feature_i, threshold)
                            best_sets = BestSets(
                                Xy1[:,1:n_features],
                                Xy1[:,end],
                                Xy2[:,1:n_features],
                                Xy2[:,end]
                            )
                        end
                    end
                end
            end
        end

        if largest_impurity > tree_settings.min_impurity
            # Build subtrees for the right and left branches
            true_branch = build_tree(
                Dataset(
                    best_sets.leftX,
                    best_sets.lefty
                ),
                current_depth + 1
            )
            false_branch = build_tree(
                Dataset(
                    best_sets.rightX,
                    best_sets.righty
                ),
                current_depth + 1
            )
            return DecisionNode(
                best_criteria.feature_i,
                best_criteria.threshold,
                missing,
                true_branch,
                false_branch
            )
        end
        # We're at leaf => determine value
        leaf_value = leaf_value_calculation(data.y)
        return DecisionNode(
            missing,
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
        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        feature_value_type = typeof(feature_value)

        if feature_value >= tree.threshold
            branch = tree.true_branch
        elseif feature_value == tree.threshold
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
