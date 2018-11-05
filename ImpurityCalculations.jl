module ImpurityCalculations
using Statistics
export calculate_variance_reduction, calculate_information_gain, gain_by_taylor
import ..Structures: NumericList

function calculate_variance_reduction(
    y::NumericList,
    y1::NumericList,
    y2::NumericList)::Float64
    var_tot = var(y)
    var_1 = var(y1)
    var_2 = var(y2)
    frac_1 = length(y1) / length(y)
    frac_2 = length(y2) / length(y)
    var_tot - ((frac_1 * var_1) + (frac_2 * var_2))
end

function calculate_entropy(y::NumericList)::Float64
    unique_labels = unique(y)
    entropy = 0
    for label in unique_labels
        count = length(y[y .== label])
        p = count / length(y)
        entropy += -p * log2(p)
    end
    entropy
end

function calculate_information_gain(
    y::NumericList,
    y1::NumericList,
    y2::NumericList)::Float64
    p = length(y1) / length(y)
    y_entropy = calculate_entropy(y)
    y1_entropy = calculate_entropy(y1)
    y2_entropy = calculate_entropy(y2)
    y_entropy - p * y1_entropy - (1 - p) * y2_entropy
end

function split(y)
    col = size(y, 2) ÷ 2
    y[:,1:col], y[:,col+1:end]
end

function gain_by_taylor(loss)
    function calculate_gain(y, y1, y2)
        function gain(y, loss)
            y, y_pred = split(y)
            nominator = sum(y .* loss.gradient(y, y_pred)) ^ 2
            denominator = sum(loss.hess(y, y_pred))
            0.5 * nominator / denominator
        end
        true_gain = gain(y1, loss)
        false_gain = gain(y2, loss)
        gain = gain(y, loss)
        true_gain + false_gain - gain
    end
    calculate_gain
end


end
