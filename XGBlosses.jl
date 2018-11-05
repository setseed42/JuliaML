module XGBlosses
export XGBloss, logisticloss

struct XGBloss
    loss
    gradient
    hess
end

function logisticloss()
    sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_grad(x) = x * (1 - x)
    function loss(y, y_pred)
        y_pred = clamp.(y_pred, 0., 1.)
        p = sigmoid.(y_pred)
        y .* log.(p) + (1 .- y) .* log.(1 .- p)
    end
    function gradient(y, y_pred)
        p = sigmoid.(y_pred)
        println(y)
        println(p)
        -(y - p)
    end
    function hess(y, y_pred)
        p = sigmoid.(y_pred)
        p .* (1 .- p)
    end
    XGBloss(loss, gradient, hess)
end




end
