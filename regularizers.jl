module Regularizers
using LinearAlgebra
export l1_regularization, l2_regularization, l1_l2_regularization


function l1_regularization(alpha::Any)
    reg(w::Any) = alpha * norm(w)
    reg_gradient(w::Any) = alpha * map(sign, w)
    return reg, reg_gradient
end


function l2_regularization(alpha::Any)
    reg(w::Any) = alpha * 0.5 * dot(transpose(w), w)
    reg_gradient(w::Any) = alpha * w
    return reg, reg_gradient
end


function l1_l2_regularization(alpha, l1_ratio)
    function reg(w)
        l1_contr = l1_ratio *norm(w)
        l2_contr = (1 - l1_ratio) * 0.5 * dot(transpose(w), w)
        return alpha * (l1_contr + l2_contr)
    end

    function reg_gradient(w)
        l1_contr = l1_ratio * map(sign, w)
        l2_contr = (1 - l1_ratio) * w
    end
    return reg, reg_gradient
end


end
