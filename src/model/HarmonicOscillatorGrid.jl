# Midpoint quadrature and real harmonic-oscillator orbitals.

"""
    ho_1d_basis(u, n_max)

Evaluate the dimensionless 1D harmonic-oscillator wavefunctions
`ψ_n(u) = π^{-1/4} (2^n n!)^{-1/2} H_n(u) exp(-u²/2)` for `n = 0..n_max` on
the grid `u`, using the factorial-free normalized recurrence

    ψ_0(u) = π^{-1/4} exp(-u²/2)
    ψ_1(u) = √2 · u · ψ_0(u)
    ψ_{n+1}(u) = √(2/(n+1)) · u · ψ_n(u) - √(n/(n+1)) · ψ_{n-1}(u)

This avoids overflow of n! for any n we care about and is numerically stable
for the modest `n ≤ 3` required by the  baseline. The 1D wavefunctions
are normalized as `∫_{-∞}^{∞} ψ_n(u) ψ_m(u) du = δ_{nm}`; the discretized
inner product on the centered grid converges to that exponentially as `L`
grows.

Returns a `Matrix{Float64}` of shape `(length(u), n_max+1)` whose column
`n+1` is `ψ_n(u_i)`.
"""
function ho_1d_basis(u::AbstractVector{<:Real}, n_max::Integer)
    n_max >= 0 || throw(ArgumentError("n_max must be ≥ 0, got $n_max"))
    N = length(u)
    Ψ = Matrix{Float64}(undef, N, n_max + 1)
    pi_quarter_inv = 1 / π^(1/4)
    @inbounds for i in 1:N
        Ψ[i, 1] = pi_quarter_inv * exp(-u[i]^2 / 2)
    end
    if n_max >= 1
        sq2 = sqrt(2.0)
        @inbounds for i in 1:N
            Ψ[i, 2] = sq2 * u[i] * Ψ[i, 1]
        end
    end
    for n in 2:n_max
        a = sqrt(2.0 / n)
        b = sqrt((n - 1) / n)
        @inbounds for i in 1:N
            Ψ[i, n+1] = a * u[i] * Ψ[i, n] - b * Ψ[i, n-1]
        end
    end
    return Ψ
end

"""
    build_orbitals_2d(grid, orbitals) -> (u, du, Φ)

Build the full 2D harmonic-oscillator orbital array on the centered midpoint
grid in `grid::GridParams`, for the deterministic orbital list in
`orbitals::OrbitalParams`.

- `u::Vector{Float64}`        — 1D grid nodes from `grid_nodes(grid)`
- `du::Float64`               — grid spacing `L / n_grid`
- `Φ::Array{Float64,3}`       — shape `(n_grid, n_grid, n_orb)`, with
                                `Φ[i, j, a] = ψ_{n_x^(a)}(u_i) ψ_{n_y^(a)}(u_j)`
                                for orbital `a = 1..n_orb`.

The 1D basis is computed once up to the maximum quantum number that appears
in `orbitals.quanta`, then tensored.
"""
function build_orbitals_2d(grid::GridParams, orbitals::OrbitalParams)
    u  = grid_nodes(grid)
    du = grid_step(grid)
    if orbitals.n_orb == 0
        return u, du, zeros(Float64, grid.n_grid, grid.n_grid, 0)
    end
    n_max = maximum(max(q[1], q[2]) for q in orbitals.quanta)
    Ψ = ho_1d_basis(u, n_max)
    N = grid.n_grid
    Φ = Array{Float64,3}(undef, N, N, orbitals.n_orb)
    @inbounds for a in 1:orbitals.n_orb
        nx, ny = orbitals.quanta[a]
        col_x = view(Ψ, :, nx + 1)
        col_y = view(Ψ, :, ny + 1)
        for j in 1:N, i in 1:N
            Φ[i, j, a] = col_x[i] * col_y[j]
        end
    end
    return u, du, Φ
end
