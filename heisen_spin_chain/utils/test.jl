pushed_vec = [1.0, 0., 0., 2.0, 0., 0., 1, 1, 0]
pushed_vec = map(normalize, [pushed_vec[3i-2:3i] for i in 1:div(length(pushed_vec), 3)])

pushed_vec