# parallel_loop.jl
using Distributed

# Add workers from the command line args (Slurm will launch N processes)
# These workers are added automatically if you use `srun julia -p N`

@everywhere function do_work(i)
    sleep(rand())  # Simulate some work
    println("Task $i done on worker $(myid())")
end

@sync @distributed for i in 1:100
    do_work(i)
end