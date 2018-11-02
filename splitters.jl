module Splitters
export train_test_split
import ..Structures: Dataset
using Random

function train_test_split(
    data::Dataset,
    split::Float64,
    reshuffle::Bool=true
)::Tuple{Dataset,Dataset}

    n_obs = size(data.x, 1)
    split_index = Int(floor(n_obs*split))
    indexes = 1:n_obs

    if reshuffle
        indexes = shuffle(indexes)
    end

    x_train = data.x[1:split_index-1, :]
    x_test = data.x[split_index:end, :]
    y_train = data.y[1:split_index-1]
    y_test = data.y[split_index:end]

    Dataset(x_train, y_train), Dataset(x_test, y_test)
end


end
