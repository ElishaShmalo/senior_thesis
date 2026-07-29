using FFTW
using Plots
using LaTeXStrings

"""
    to_komega(data; dt=1.0, dx=1.0, logscale=true, subtract_mean=true)

Take a real-space / real-time signal `data[t, j]` (rows = time t, columns = site j,
which is the layout your `hcat(delta_spins...)'` produces) and return its power
spectrum in (k, ω) space along with the matching axis vectors.

Returns `(S, ks, ωs)` where:
  - `S`   is the (shifted) spectrum, `S[ω_index, k_index]`
  - `ks`  is the momentum axis (length = number of sites)    — angular wavenumber
  - `ωs`  is the frequency axis (length = number of times)   — angular frequency

Notes
  - `dx` is the spatial lattice spacing (1 site = 1 here, so dx = 1).
  - `dt` is the time step *between successive rows* of your data — set this to
    whatever the sampling interval of your evolution actually is.
  - `subtract_mean` removes the DC (k=0, ω=0) component so it doesn't swamp the
    color scale (the uniform background is rarely the interesting part).
  - `logscale` returns log10 of the power, which is almost always what you want
    for a k-ω plot since the dynamic range is huge.
"""
function to_komega(data::AbstractMatrix; dt::Real=1.0, dx::Real=1.0,
                   logscale::Bool=true, subtract_mean::Bool=true)
    A = Float64.(data)
    if subtract_mean
        A = A .- sum(A) / length(A)
    end

    Nt, Nj = size(A)              # rows = time, cols = space

    # 2D FFT: over time (dim 1) and space (dim 2)
    F = fftshift(fft(A))

    # Power spectrum, with the zero-frequency component moved to the center
    # P = abs2.(fftshift(F))
    P = abs2.(F)

    # Conjugate axes. fftfreq returns ordinary frequency (cycles/unit);
    # multiply by 2π for angular wavenumber / angular frequency.
    ωs = 2π .* fftshift(fftfreq(Nt, 1 / dt))   # energy / frequency axis
    ks = 2π .* fftshift(fftfreq(Nj, 1 / dx))   # momentum axis

    S = logscale ? log10.(P .+ eps()) : P
    return S, ks, ωs
end

function fft_komega(data::AbstractMatrix; dt::Real=1.0, dx::Real=1.0)
    A = Float64.(data)
    Nt, Nj = size(A)
    F  = fftshift(fft(A))
    ωs = 2π .* fftshift(fftfreq(Nt, 1 / dt))
    ks = 2π .* fftshift(fftfreq(Nj, 1 / dx))
    return F, ks, ωs
end

function ifft_komega(F::AbstractMatrix; dj::Real=1.0, dt::Real=1.0)
    A = F
    Nt, Nj = size(A)
    D  = ifft(ifftshift(A))
    ts = (0:Nt-1) .* dt
    js = (0:Nj-1) .* dj
    return D, js, ts
end

function verify_roundtrip(data::AbstractMatrix; dt::Real=1.0, dx::Real=1.0,
                          verbose::Bool=true)
    A = Float64.(data)
    F, _, _   = fft_komega(A; dt=dt, dx=dx)
    recon_c   = ifft(ifftshift(F))          # keep complex to inspect imag part
    max_err   = maximum(abs.(real.(recon_c) .- A))
    max_imag  = maximum(abs.(imag.(recon_c)))
    if verbose
        println("round-trip max |Δ|      = ", max_err)
        println("round-trip max |Im|     = ", max_imag)
        println(max_err < 1e-8 ? "✓ reconstruction OK" : "✗ reconstruction FAILED")
    end
    return max_err, max_imag
end