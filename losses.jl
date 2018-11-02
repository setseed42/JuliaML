module Losses
using Statistics
import ..Structures: Dataset, Regularizer, Loss
export mse

function mse(data::Dataset, w::Array{Float64,1}, regularizer::Regularizer)::Loss
    y_pred = data.x * w
    reg_weights = append!(w[1:end-1], 0)
    mse = mean(0.5 * (data.y - y_pred).^2) .+ regularizer.reg(reg_weights)
    grad_w = -transpose(data.x) * (data.y - y_pred) + regularizer.reg_gradient(reg_weights)
    return Loss(mse, grad_w)
end

end
