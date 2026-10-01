const CORNER_HERE = @__DIR__
const CORNER_LAMB_DIR = abspath(joinpath(CORNER_HERE, "..", "..", ".."))

include(joinpath(CORNER_LAMB_DIR, "LambDickeModel.jl"))
include(joinpath(CORNER_LAMB_DIR, "HarmonicOscillatorGrid.jl"))
include(joinpath(CORNER_LAMB_DIR, "Speckle.jl"))
include(joinpath(CORNER_LAMB_DIR, "OverlapIntegrals.jl"))
include(joinpath(CORNER_LAMB_DIR, "IntegrableCorner.jl"))

using .IntegrableCorner

const CORNER_N_GRID = 160
const CORNER_BOX_LENGTH = 15.0
const CORNER_CORRELATION_LENGTH = 1.0
const CORNER_DISORDER_STRENGTH = 1.0
const CORNER_J = 1.0
const CORNER_GAMMA_EFF = 1.0
const CORNER_KAPPA_EFF = 0.2
const CORNER_GAMMA_TOT = CORNER_GAMMA_EFF + CORNER_KAPPA_EFF
const CORNER_EPSILON_ZERO = 1.0e-8
const CORNER_N_BINS = 40
const CORNER_DEGREE = 5
const CORNER_TAU_RANGE = (1.0e-3, 10.0, 250)

function build_physical_corner_g(n_orb::Integer, seed::Integer;
                                 n_grid::Integer = CORNER_N_GRID,
                                 box_length::Real = CORNER_BOX_LENGTH,
                                 correlation_length::Real = CORNER_CORRELATION_LENGTH,
                                 disorder_strength::Real = CORNER_DISORDER_STRENGTH)
    params = SingleParticleParams(
        grid = GridParams(box_length = box_length, n_grid = n_grid),
        orbitals = OrbitalParams(n_orb = n_orb),
        weight = WeightParams(
            weight_type = :speckle,
            speckle = SpeckleParams(
                correlation_length = correlation_length,
                disorder_strength = disorder_strength,
                seed = seed,
            ),
        ),
    )
    coefficients = compute_coefficients(params)
    return coefficients.g_jk
end

corner_tau_grid(n::Integer = CORNER_TAU_RANGE[3]) =
    exp.(range(log(CORNER_TAU_RANGE[1]), log(CORNER_TAU_RANGE[2]); length = Int(n)))
