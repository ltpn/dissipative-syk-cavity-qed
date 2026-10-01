# Speckle intensity and detuning weights.

using FFTW: fftfreq, ifft
using Random: MersenneTwister
using Statistics: mean

"""
    speckle_intensity(grid, speckle) -> Matrix{Float64}

Generate the normalized speckle intensity `I(u)` on the 2D grid described by
`grid::GridParams` for the parameters in `speckle::SpeckleParams`.

Procedure (locked):

1. Build the 2D angular wavenumber grid
   `k_x[i] = 2π · fftfreq(N, N/L)[i]`, similarly `k_y`.
2. Sample iid complex-Gaussian coefficients `c[i, j]` for every mode with
   `k_x[i]² + k_y[j]² ≤ k_c²`, where `k_c = 2π / ξ`. Modes outside the disk
   are zero.
3. `Ẽ = ifft(c)`; `Ĩ = |Ẽ|²`; `I = Ĩ / mean(Ĩ)`.

The output `I` has `⟨I⟩ = 1` exactly (in floating-point arithmetic this means
`abs(mean(I) - 1)` is at the level of round-off, ≪ 1e-12) and is everywhere
non-negative.

Reproducibility: the RNG is `MersenneTwister(speckle.seed)`, seeded once per
realization. The same seed yields the same intensity field for the same
`(L, N, ξ)`.
"""
function speckle_intensity(grid::GridParams, speckle::SpeckleParams)
    L = grid.box_length
    N = grid.n_grid
    ξ = speckle.correlation_length

    rng = MersenneTwister(speckle.seed)

    # Angular wavenumber grid (radians per unit u).  fftfreq returns
    # frequencies in cycles per unit; multiply by 2π to get rad/unit.
    f1d = fftfreq(N, N / L)
    k = 2π .* collect(f1d)

    k_c2 = (2π / ξ)^2

    coeffs = zeros(ComplexF64, N, N)
    @inbounds for j in 1:N, i in 1:N
        if k[i]^2 + k[j]^2 <= k_c2
            coeffs[i, j] = randn(rng, ComplexF64)
        end
    end

    # Edge case: empty disk (ξ smaller than the largest representable mode);
    # we still want a well-defined output.  This cannot happen for the
    # baseline ξ = 1, L = 15: the disk contains O(L²/ξ²) ≈ 225 modes.  It is
    # possible only for absurd `ξ` choices.
    if all(iszero, coeffs)
        # Force a single nonzero (deterministic-on-seed) DC mode to avoid a
        # 0/0 in the normalization step.  The mean of the resulting I is 1.
        coeffs[1, 1] = ComplexF64(1.0)
    end

    # Force FFTW for this (generally non-power-of-2) N x N transform.  When
    # AppleAccelerate is loaded it pirates `AbstractFFTs.plan_bfft` for `Array`
    # inputs and routes them to vDSP, which only supports power-of-2 sizes
    # (e.g. n_grid = 160 fails).  Passing a `view` (a `SubArray`, not an
    # `Array`) dodges that `Array`-only piracy so the transform stays on FFTW,
    # while AppleAccelerate's BLAS acceleration is kept.  On non-Apple platforms
    # FFTW handles the `SubArray` identically, so the result is unchanged.
    E = ifft(@view coeffs[:, :])
    intensity_unnorm = abs2.(E)
    m = mean(intensity_unnorm)
    intensity_unnorm ./ m
end

"""
    weight_field(I, f) -> Matrix{Float64}

Return `w(u) = 1 / (1 + f · I(u))` element-wise. With `I` non-negative and
`f ≥ 0`, the result is in `(0, 1]`. Uniform control: pass `I = zeros(...)`
and any `f`, or call `weight_field_uniform(grid)` directly.
"""
function weight_field(I::AbstractMatrix{<:Real}, f::Real)
    return @. 1.0 / (1.0 + f * I)
end

"""
    weight_field_uniform(grid::GridParams) -> Matrix{Float64}

Return the uniform-weight control `w(u) ≡ 1` on the grid. Used by the
:uniform branch of the `Coefficients` builder.
"""
function weight_field_uniform(grid::GridParams)
    return ones(Float64, grid.n_grid, grid.n_grid)
end

function build_weight(grid::GridParams, weight::WeightParams)
    if weight.weight_type == :uniform
        return weight_field_uniform(grid)
    elseif weight.weight_type == :speckle
        sp = weight.speckle
        sp === nothing && throw(ArgumentError(
            "weight_type=:speckle requires SpeckleParams"))
        I = speckle_intensity(grid, sp)
        return weight_field(I, sp.disorder_strength)
    else
        throw(ArgumentError("Unknown weight_type=$(weight.weight_type)"))
    end
end
