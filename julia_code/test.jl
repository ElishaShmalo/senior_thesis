include("utils/general.jl")


L = 256

k=1
B=1
phi_j_coff = 1

s_js = [B * tanh(k*j) for j in -round(L/2):round(L/2)][1:L]
phi_js = [phi_j_coff * B * sech(k*j) / sqrt(2) for j in -round(L/2):round(L/2)][1:L]

ts = [j for j in 1:L]
plt = plot()
plot!(ts, s_js, label="s")
plot!(ts, phi_js, label="phi")

display(plt)
