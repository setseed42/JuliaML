module Splitters
export train_test_split, get_random_subset
import ..Structures: Dataset
using Random

function train_test_split(
    data::Dataset,
    split::Float64,
    reshuffle::Bool=true
)::Tuple{Dataset,Dataset}

    n_obs = size(data.x, 1)
    split_index = floor(Int64, n_obs*split)
    indexes = 1:n_obs

    if reshuffle
        indexes = shuffle(indexes)
    end
    train_indexes = indexes[1:split_index-1]
    test_indexes = indexes[split_index:end]
    x_train = data.x[train_indexes, :]
    x_test = data.x[test_indexes, :]
    y_train = data.y[train_indexes]
    y_test = data.y[test_indexes]

    Dataset(x_train, y_train), Dataset(x_test, y_test)
end

function get_random_subset(
    data::Dataset,
    subsample_size::Float64
    )::Dataset
    n_obs = size(data.x, 1)
    split_ix = floor(Int64, n_obs * subsample_size)
    indexes = randperm(n_obs)[1:split_ix]
    Dataset(data.x[indexes,:], data.y[indexes])
end


end
