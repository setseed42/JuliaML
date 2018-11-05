module LeafValueCalculations
using Statistics
import ..Structures: NumericList
export mean_of_y, majority_vote, approximate_update

function mean_of_y(y::NumericList)::Float64
    mean(y)
end

function majority_vote(y::NumericList)::Union{Float64,Int64}
    most_common = missing
    max_count = 0
    for label in unique(y)
        count = length(y[y .== label])
        if count > max_count
            most_common = label
            max_count = count
        end
    end
    most_common
end

function split(y)
    col = size(y, 2) ÷ 2
    y[:,1:col], y[:,col+1:end]
end

function approximate_update(loss)
    function calculate_update(y)
        y, y_pred = split(y)
        gradient = sum(y .* loss.gradient(y, y_pred), dims=1)
        hessian = sum(loss.hess(y, y_pred), dims=1)
        gradient ./ hessian
    end
    calculate_update
end


end
