module Structures

export Dataset, Regularizer, Loss

struct Dataset
    x::Array{Float64}
    y::Array{Float64, 1}
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
