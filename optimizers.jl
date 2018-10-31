module Optimizers
using Distributions
using Random
export gradient_descent, minibatch_gradient_descent


function initialize_weights(n_features)
    limit = 1 / sqrt(n_features)
    distribution = Uniform(-limit, limit)
    w = rand(distribution, n_features)
    return w
end


function early_stopper(curr, best, early_stopping)
    i, loss, w = curr
    best_i, best_loss, best_w = best
    if loss < best_loss
        finished = false
        return curr, finished
    end
    if i - best_i > early_stopping
        finished = true
        return best, finished
    end
    finished = false
    return best, finished
end


function make_minibatch(n_obs, batch_size)
    n_batches = n_obs/batch_size
    indexes = shuffle(1:n_obs)
    batched_indexes = []
    batch = Int64[]
    for i = indexes
        append!(batch, i)
        if length(batch) >= batch_size
            push!(batched_indexes, batch)
            batch = Int64[]
        end
    end
    if length(batch) > 0
        push!(batched_indexes, batch)
    end
    return batched_indexes
end


function minibatch_gradient_descent(early_stopping, learning_rate, batch_size)
    function optimizer(x_train, y_train, x_test, y_test, regularizer, loss_fn)
        i = 1
        n_features = size(x_train, 2)
        w = initialize_weights(n_features)
        reg, reg_gradient = regularizer

        best = i, Inf, w
        finished = false
        batched_indexes = make_minibatch(size(x_train, 2), batch_size)
        while !finished
            for batch_index = batched_indexes
                x_batch = x_train[batch_index,:]
                y_batch = y_train[batch_index]
                batch_loss, grad_w = loss_fn(x_batch, y_batch, w, reg, reg_gradient)
                w = w - learning_rate * grad_w
            end
            test_loss, _ = loss_fn(x_test, y_test, w, reg, reg_gradient)
            curr = i, test_loss, w
            best, finished = early_stopper(curr, best, early_stopping)

            if finished
                best_i, best_loss, best_w = best
                global best_w = best_w
                println("Finished at iteration: $i with best test loss: $best_loss at iteration: $best_i")
                break
            end
            i += 1
        end
        return best_w
    end
    return optimizer
end


function gradient_descent(early_stopping, learning_rate)
    function optimizer(x_train, y_train, x_test, y_test, regularizer, loss_fn)
        i = 1
        n_features = size(x_train, 2)
        w = initialize_weights(n_features)
        reg, reg_gradient = regularizer

        best = i, Inf, w
        finished = false
        while !finished
            train_loss, grad_w = loss_fn(x_train, y_train, w, reg, reg_gradient)
            w = w - learning_rate * grad_w
            test_loss, _ = loss_fn(x_test, y_test, w, reg, reg_gradient)
            curr = i, test_loss, w
            best, finished = early_stopper(curr, best, early_stopping)
            if finished
                best_i, best_loss, best_w = best
                global best_w = best_w
                println("Finished at iteration: $i with best test loss: $best_loss at iteration: $best_i")
                break
            end
            i += 1
        end
        return best_w
    end
    return optimizer
end


end
