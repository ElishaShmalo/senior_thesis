# Part C of cluster_launch_test.sh: the real launch path (SlurmClusterManager + addprocs), timed,
# with a 15-minute launch timeout and --startup-file=no for the workers.
using Distributed, SlurmClusterManager
t0 = time()
el() = round(time() - t0; digits=1)
println("C: addprocs(SlurmManager) starting, expecting $(ENV["SLURM_NTASKS"]) workers"); flush(stdout)
try
    addprocs(SlurmManager(launch_timeout=900.0); exeflags="--startup-file=no")
catch e
    println("C: addprocs FAILED after $(el()) s with $(nprocs() - 1) workers up")
    println("C: ", first(sprint(showerror, e), 400))
    flush(stdout)
    exit(1)
end
println("C: addprocs OK after $(el()) s: $(nworkers()) workers"); flush(stdout)
hosts = [remotecall_fetch(gethostname, w) for w in workers()]
counts = Dict{String,Int}()
for h in hosts
    counts[h] = get(counts, h, 0) + 1
end
for (h, n) in sort(collect(counts))
    println("C:   $n workers on $h")
end
println("C: round trip to every worker done after $(el()) s"); flush(stdout)
