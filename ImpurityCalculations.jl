module ImpurityCalculations
using Statistics
export calculate_variance_reduction, calculate_information_gain

function safe_var(x::Array{Float64})::Float64
    variance = var(x)
    if isnan(variance)
        variance = 0
    end
    variance
end

function calculate_variance_reduction(y::Array{Float64}, y1::Array{Float64}, y2::Array{Float64})::Float64
    var_tot = safe_var(y)
    var_1 = safe_var(y1)
    var_2 = safe_var(y2)
    frac_1 = length(y1) / length(y)
    frac_2 = length(y2) / length(y)
    variance_reduction = var_tot - ((frac_1 .* var_1) + (frac_2 .* var_2))
    sum(variance_reduction)
end

function calculate_entropy(y::Union{Array{Float64},Array{Int64}})::Float64
    unique_labels = unique(y)
    entropy = 0
    for label in unique_labels
        count = length(y[y .== label])
        p = count / length(y)
        entropy += -p * log2(p)
    end
    entropy
end

function calculate_information_gain(y::Union{Array{Float64},Array{Int64}}, y1::Array{Float64}, y2::Array{Float64})::Float64
    p = length(y1) / length(y)
    y_entropy = calculate_entropy(y)
    y1_entropy = calculate_entropy(y1)
    y2_entropy = calculate_entropy(y2)
    y_entropy - p * y1_entropy - (1 - p) * y2_entropy
end

end
