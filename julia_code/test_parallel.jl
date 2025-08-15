using Distributed

@everywhere begin
    println("Hello from worker $(myid()) on host $(gethostname())")
end

@everywhere begin
    # Code in this block runs on all workers
    function heavy_computation(x)
        # Simulate some heavy work
        sleep(rand())  # Sleep for random seconds to simulate workload
        println(x, x^2)
    end
end

# Create a range of numbers to process
numbers = 1:128

println("Running parallel computation on ", nworkers(), " workers...")

# Run the computation in parallel
@sync @distributed for x in numbers
    heavy_computation(x)
end

println("Done:")



