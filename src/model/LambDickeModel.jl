# Grid, orbital, disorder, and single-particle coefficient types.

using LinearAlgebra: norm

"""
    GridParams(; box_length, n_grid)

Centered midpoint Cartesian grid in dimensionless coordinates `u = r/x0`.

- `box_length::Float64`  — `L / x0` (full extent of the grid in `u`).
- `n_grid::Int`          — number of points per side (even or odd).

The grid nodes are
`u_i = -L/2 + Δu * (i - 1/2)` for `i = 1..n_grid`, with `Δu = L / n_grid`.
This is the midpoint quadrature convention.
"""
struct GridParams
    box_length::Float64
    n_grid::Int
end

GridParams(; box_length::Real, n_grid::Integer) =
    GridParams(Float64(box_length), Int(n_grid))

"""
    OrbitalParams(; n_orb)

Deterministic 2D harmonic-oscillator orbital list at half filling.

Orbitals are indexed `a = 1..n_orb` and labelled by quantum numbers
`(n_x, n_y)` ordered by ascending shell `s = n_x + n_y` and within a shell by
ascending `n_x`. The tie-break "then ascending `n_y`" from
`docs/model.md` is automatic because `n_y = s - n_x` is
determined inside a shell.

`n_orb` must be even (half-filling block needs `N_f = n_orb/2` particles).
"""
struct OrbitalParams
    n_orb::Int
    quanta::Vector{Tuple{Int,Int}}  # (n_x, n_y) for each orbital
end

"""
    orbital_quanta(n_orb)

Return the deterministic (n_x, n_y) list for the first `n_orb` 2D HO orbitals
with the ordering defined in `docs/model.md`.
"""
function orbital_quanta(n_orb::Integer)
    n_orb >= 0 || throw(ArgumentError("n_orb must be non-negative, got $n_orb"))
    out = Vector{Tuple{Int,Int}}()
    sizehint!(out, n_orb)
    s = 0
    while length(out) < n_orb
        for nx in 0:s
            ny = s - nx
            push!(out, (nx, ny))
            length(out) >= n_orb && return out
        end
        s += 1
    end
    return out  # unreachable
end

OrbitalParams(; n_orb::Integer) = OrbitalParams(Int(n_orb), orbital_quanta(n_orb))

"""
    SpeckleParams(; correlation_length, disorder_strength, seed)

Speckle generation parameters in dimensionless units.

- `correlation_length::Float64`   — `ξ / x0`. Sets the Fourier cutoff
                                    `k_c = 2π / ξ`.
- `disorder_strength::Float64`    — `f` in `w = 1 / (1 + f I)`.
- `seed::Int`                      — RNG seed for reproducibility.
"""
struct SpeckleParams
    correlation_length::Float64
    disorder_strength::Float64
    seed::Int
end

SpeckleParams(; correlation_length::Real, disorder_strength::Real, seed::Integer) =
    SpeckleParams(Float64(correlation_length), Float64(disorder_strength), Int(seed))

"Uniform or speckle disorder weight."
struct WeightParams
    weight_type::Symbol
    speckle::Union{SpeckleParams, Nothing}
end
WeightParams(; weight_type::Symbol = :uniform,
              speckle::Union{SpeckleParams, Nothing} = nothing) =
    WeightParams(weight_type, speckle)

"Single-particle grid, orbital, and disorder parameters."
struct SingleParticleParams
    grid::GridParams
    orbitals::OrbitalParams
    weight::WeightParams
end

SingleParticleParams(; grid::GridParams, orbitals::OrbitalParams,
                        weight::WeightParams) =
    SingleParticleParams(grid, orbitals, weight)

"""
    Coefficients(K_jk, g_jk, X_jk, T_jk, grid, orbitals, weight)

Single-particle overlap matrices required by  and 5 (symbol table
`docs/model.md`):

- `K_jk[a, b] = ∫ d²u w(u) φ_a(u) φ_b(u)`              real symmetric
- `g_jk[a, b] = ∫ d²u g_d(u) g(u) w(u) φ_a φ_b`        real symmetric;
                                                       baseline `g = g_d = 1` ⇒ `g_jk = K_jk`
- `X_jk[axis][a, b] = ∫ d²u u_axis w φ_a φ_b`,
   `axis ∈ {1=:x, 2=:y}`                                real symmetric
- `T_jk[a, b] = ∫ d²u |u|² w φ_a φ_b`                   real symmetric

`grid`, `orbitals`, `weight` echo the parameters used to build the matrices.
"""
struct Coefficients
    K_jk::Matrix{Float64}
    g_jk::Matrix{Float64}
    X_jk::NTuple{2, Matrix{Float64}}
    T_jk::Matrix{Float64}
    grid::GridParams
    orbitals::OrbitalParams
    weight::WeightParams
end

# ============================================================================
# Validation helpers
# ============================================================================

"""
    validate_params(p::SingleParticleParams)

Throw an informative error if any locked convention is violated. Used both at
the top of every  driver and at the start of each  module
function that takes a parameter struct.
"""
function validate_params(p::SingleParticleParams)
    p.grid.box_length > 0 ||
        throw(ArgumentError("grid.box_length must be positive, got $(p.grid.box_length)"))
    p.grid.n_grid > 0 ||
        throw(ArgumentError("grid.n_grid must be positive, got $(p.grid.n_grid)"))
    iseven(p.orbitals.n_orb) ||
        throw(ArgumentError("Half filling requires even n_orb, got $(p.orbitals.n_orb)"))
    p.orbitals.n_orb > 0 ||
        throw(ArgumentError("n_orb must be positive, got $(p.orbitals.n_orb)"))
    length(p.orbitals.quanta) == p.orbitals.n_orb ||
        throw(ArgumentError("orbital quanta length mismatch"))
    # Verify deterministic ordering: ascending shell, ascending n_x within shell.
    for a in 2:p.orbitals.n_orb
        (nx_p, ny_p) = p.orbitals.quanta[a-1]
        (nx_c, ny_c) = p.orbitals.quanta[a]
        s_p, s_c = nx_p + ny_p, nx_c + ny_c
        ok = s_c > s_p || (s_c == s_p && nx_c > nx_p)
        ok || throw(ArgumentError(
            "orbital ordering violated at a=$a: ($nx_p,$ny_p) → ($nx_c,$ny_c)"))
    end
    p.weight.weight_type in (:uniform, :speckle) ||
        throw(ArgumentError("weight_type must be :uniform or :speckle, " *
                            "got $(p.weight.weight_type)"))
    if p.weight.weight_type == :speckle
        p.weight.speckle === nothing &&
            throw(ArgumentError("weight_type=:speckle requires SpeckleParams"))
        p.weight.speckle.correlation_length > 0 ||
            throw(ArgumentError("speckle correlation_length must be positive"))
        p.weight.speckle.disorder_strength >= 0 ||
            throw(ArgumentError("speckle disorder_strength must be ≥ 0"))
    end
    return nothing
end

"""
    grid_step(g::GridParams) -> Float64

Return Δu = L / n_grid.
"""
grid_step(g::GridParams) = g.box_length / g.n_grid

"""
    grid_nodes(g::GridParams) -> Vector{Float64}

Return the centered midpoint grid nodes `u_i = -L/2 + Δu (i - 1/2)`, length
`n_grid`. The 2D grid is the Cartesian product of this with itself.
"""
function grid_nodes(g::GridParams)
    L = g.box_length
    N = g.n_grid
    du = L / N
    u = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        u[i] = -L/2 + du * (i - 0.5)
    end
    return u
end
