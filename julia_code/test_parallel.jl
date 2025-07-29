# parallel_loop.jl
#using Distributed

#addprocs(Sys.CPU_THREADS - 1)  # or hardcode: addprocs(9)

#@everywhere using DistributedArrays


#@everywhere println("Worker $(myid()) running on $(gethostname())")

#my_list = distribute(zeros(10))


# @sync @distributed for i in 1:length(my_list)
#     t = rand()
#     sleep(t)  # Simulate some work
#     my_list[i] = t
#     println("Task $i done on worker $(myid())")
# end

# println(sum(my_list) / length(my_list))

using Distributed
addprocs(Sys.CPU_THREADS - 1)  # or hardcode: addprocs(9)

@everywhere using SharedArrays
@everywhere println("Hello from worker $(myid()) on $(gethostname())")

A = SharedArray{Float64}(10)
@distributed for i in 1:10
    A[i] = rand()
end
@show A
