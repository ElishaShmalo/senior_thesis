using DifferentialEquations
using StaticArrays
using LinearAlgebra  # for cross
using Plots

# Define dynamics: u, v are 3D vectors (SVector)
function cross_dynamics!(du, u, p, t)
    u_vec = @SVector [u[1], u[2], u[3]]
    v_vec = @SVector [u[4], u[5], u[6]]

    # Example dynamics using cross product
    du_vec = cross(u_vec, v_vec)  # du/dt = u × v
    dv_vec = -cross(u_vec, v_vec) # dv/dt = -u × v

    du[1:3] = du_vec
    du[4:6] = dv_vec
end

# Initial vectors u and v
u0 = [1.0, 0.0, 0.0,   # u
      0.0, 1.0, 0.0]   # v
tspan = (0.0, 10.0)

prob = ODEProblem(cross_dynamics!, u0, tspan)
sol = solve(prob, RK4(), dt=0.01)

# Extract time series
u_x = [u[1] for u in sol.u]
u_y = [u[2] for u in sol.u]
u_z = [u[3] for u in sol.u]

plot(sol.t, [u_x u_y u_z], label=["uₓ" "u_y" "u_z"],
     title="u Vector Components Over Time", xlabel="Time", ylabel="Value")



X = @SVector([@SVector[1, 2, 3] for _ in 1:4])

println(typeof(X))