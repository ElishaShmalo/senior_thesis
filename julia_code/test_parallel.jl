# parallel_loop.jl
using Distributed
addprocs(2)
@everywhere using SharedArrays


@everywhere println("Hi from $(myid())")

# Add workers from the command line args (Slurm will launch N processes)
# These workers are added automatically if you use `srun julia -p N`

my_list = SharedArray{Float64}(100)

print(my_list)

# @sync @distributed for i in 1:100
#     t = rand()
#     sleep(t)  # Simulate some work
#     my_list[i] = t
#     println("Task $i done on worker $(myid())")
# end

# println(sum(my_list) / length(my_list))
