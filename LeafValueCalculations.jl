module LeafValueCalculations
using Statistics
export mean_of_y, majority_vote

function mean_of_y(y::Array{Float64})::Array{Float64}
    mean(y, dims=1)
end

function majority_vote(y::Array{Float64})::Float64
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

end
