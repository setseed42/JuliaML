module Splitters
export train_test_split
using Random


function train_test_split(x, y, split, reshuffle=true)
    n_obs = size(x, 1)
    split_index = Int(floor(n_obs*split))
    indexes = 1:n_obs

    if reshuffle
        indexes = shuffle(indexes)
    end

    x_train = x[1:split_index-1, :]
    x_test = x[split_index:end, :]
    y_train = y[1:split_index-1]
    y_test = y[split_index:end]

    return x_train, y_train, x_test, y_test
end


end
