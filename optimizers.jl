module Optimizers
using Distributions
using Random
import ..Structures: Dataset, Loss, Regularizer
export gradient_descent, minibatch_gradient_descent


function initialize_weights(n_features::Int64)::Array{Float64}
    limit = 1 / sqrt(n_features)
    distribution = Normal(0, 1)
    w = rand(distribution, n_features)
    return w
end

struct IterInfo
    iteration::Int64
    loss::Float64
    w::Array{Float64}
end


function early_stopper(curr::IterInfo, best::IterInfo, early_stopping::Int64)::Tuple{IterInfo, Bool}
    if curr.loss < best.loss
        finished = false
        return curr, finished
    end
    if curr.iteration - best.iteration > early_stopping
        finished = true
        return best, finished
    end
    finished = false
    return best, finished
end


function gradient_descent(early_stopping::Int64, learning_rate::Float64)
    function optimizer(train::Dataset, test::Dataset, regularizer::Regularizer, loss_fn)::Array{Float64}
        i = 1
        n_features = size(train.x, 2)
        w = initialize_weights(n_features)

        best = IterInfo(i, Inf, w)
        finished = false
        while !finished
            train_loss = loss_fn(train, w, regularizer)
            w = w - learning_rate * train_loss.w_grad
            test_loss = loss_fn(test, w, regularizer)
            curr = IterInfo(i, test_loss.loss, w)
            best, finished = early_stopper(curr, best, early_stopping)
            if finished
                best_i, best_loss, best_w = best.iteration, best.loss, best.w
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


# function make_minibatch(n_obs, batch_size)
#     n_batches = n_obs/batch_size
#     indexes = shuffle(1:n_obs)
#     batched_indexes = []
#     batch = Int64[]
#     for i = indexes
#         append!(batch, i)
#         if length(batch) >= batch_size
#             push!(batched_indexes, batch)
#             batch = Int64[]
#         end
#     end
#     if length(batch) > 0
#         push!(batched_indexes, batch)
#     end
#     return batched_indexes
# end


# function minibatch_gradient_descent(early_stopping, learning_rate, batch_size)
#     function optimizer(train, test, regularizer, loss_fn)
#         i = 1
#         n_features = size(train.x, 2)
#         w = initialize_weights(n_features)

#         best = i, Inf, w
#         finished = false
#         batched_indexes = make_minibatch(size(train.x, 2), batch_size)
#         batched_data = map(indexes -> Dataset(train.x[indexes,:], train.y[indexes]), batched_indexes)
#         while !finished
#             for batch = batched_data
#                 batch_loss = loss_fn(batch, w, regularizer)
#                 w = w - learning_rate * batch_loss.w_grad
#             end
#             test_loss = loss_fn(test, w, regularizer)
#             curr = IterInfo(i, test_loss.loss, w)
#             best, finished = early_stopper(curr, best, early_stopping)

#             if finished
#                 best_i, best_loss, best_w = best.iteration, best.loss, best.w
#                 global best_w = best.w
#                 println("Finished at iteration: $i with best test loss: $best_loss at iteration: $best_i")
#                 break
#             end
#             i += 1
#         end
#         return best_w
#     end
#     return optimizer
# end

end
