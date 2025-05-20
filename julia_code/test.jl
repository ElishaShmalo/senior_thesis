using DifferentialEquations
using Plots

# Parameters
const a = 1.1   # Prey birth rate
const b = 0.4   # Predation rate
const c = 0.4   # Predator death rate
const d = 0.1   # Predator increase from eating prey

# Lotka–Volterra ODEs
function lotka_volterra!(du, u, p, t)
    x, y = u
    du[1] = a*x - b*x*y       # dx/dt
    du[2] = d*x*y - c*y       # dy/dt
end

# Initial populations: 10 prey, 5 predators
u0 = [10.0, 5.0]
tspan = (0.0, 30.0)

# Define and solve the ODE problem
prob = ODEProblem(lotka_volterra!, u0, tspan)
sol = solve(prob, RK4(), dt=0.01)

println(sol.u)

# Plot the result
plot(sol.t, hcat(sol.u...)', xlabel="Time", ylabel="Population",
     label=["Prey" "Predator"], title="Predator-Prey Dynamics (Lotka-Volterra)")    