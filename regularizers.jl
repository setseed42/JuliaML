module Regularizers
using LinearAlgebra
export l1_regularization, l2_regularization, l1_l2_regularization
import ..Structures: Regularizer

function l1_regularization(alpha::Float64)::Regularizer
    reg(w::Array{Float64, 1})::Float64 = alpha * norm(w)
    reg_gradient(w::Array{Float64, 1})::Array{Float64, 1} = alpha * sign.(w)
    Regularizer(reg, reg_gradient)
end


function l2_regularization(alpha::Float64)::Regularizer
    reg(w::Array{Float64, 1})::Float64 = alpha * 0.5 * dot(transpose(w), w)
    reg_gradient(w::Array{Float64, 1})::Array{Float64, 1} = alpha * w
    Regularizer(reg, reg_gradient)
end


function l1_l2_regularization(alpha::Float64, l1_ratio::Float64)::Regularizer
    function reg(w::Array{Float64, 1})::Float64
        l1_contr = l1_ratio *norm(w)
        l2_contr = (1 - l1_ratio) * 0.5 * dot(transpose(w), w)
        alpha * (l1_contr + l2_contr)
    end

    function reg_gradient(w::Array{Float64, 1})::Array{Float64, 1}
        l1_contr = l1_ratio * sign.(w)
        l2_contr = (1 - l1_ratio) * w
        alpha * (l1_contr + l2_contr)
    end
    Regularizer(reg, reg_gradient)
end


end
