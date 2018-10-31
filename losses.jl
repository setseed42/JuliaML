module Losses
using Statistics
export mse

function mse(x, y, w, reg, reg_gradient)
    y_pred = x * w
    reg_weights = append!(w[1:end-1], 0)
    mse = mean(0.5 * (y - y_pred).^2) .+ reg(reg_weights)
    grad_w = -transpose(x) * (y - y_pred) + reg_gradient(reg_weights)
    return mse, grad_w
end

end
