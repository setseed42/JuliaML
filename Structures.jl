module Structures

export Dataset, Regularizer, Loss

struct Dataset
    x::Array{Float64}
    y::Union{Array{Float64, 1}, Array{Int64, 1}}
end

struct Regularizer
    reg
    reg_gradient
end

struct Loss
    loss::Float64
    w_grad::Array{Float64, 1}
end

end
