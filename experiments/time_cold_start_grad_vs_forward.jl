import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Jutul
using JutulDarcy
using MultipliersZonation
using GeoEnergyIO

# ----- Constants (no env vars / no CLI args) -----
const SPE1_WELLS = [:PROD, :INJ]
const LOSS_MODE = :rates
const SPE1_RATE_REL_FLOOR = 1e-2
const RATE_WEIGHT = 1.0
const BHP_WEIGHT = RATE_WEIGHT

# Backward-compatible wrapper for compute_auto_scales
_auto_scales(ws, wells; method = :rms) = try
    compute_auto_scales(ws, wells; method = method)
catch err
    err isa MethodError || rethrow()
    compute_auto_scales(ws, wells)
end

# ---------------- Base model setup ----------------

data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data = parse_data_file(data_pth)
case_truth = setup_case_from_data_file(data)

println("Forward simulation (cold) @time:")
result_truth = @time begin
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            simulate_reservoir(case_truth)
        end
    end
end

println("Forward simulation (hot) @time:")
_ = @time begin
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            simulate_reservoir(case_truth)
        end
    end
end

ws = result_truth.wells
step_times = cumsum(case_truth.dt)
total_time = step_times[end]

# ---------------- Loss setup ----------------

rate_scales, bhp_scale = _auto_scales(ws, SPE1_WELLS; method = :rms)
registry = default_loss_registry()
rate_obs = build_rate_observations(
    ws;
    wells = SPE1_WELLS,
    step_times = step_times,
    total_time = total_time,
    rate_scales = rate_scales,
    rate_rel_floor = SPE1_RATE_REL_FLOOR,
    rate_weight = RATE_WEIGHT,
    bhp_scale = bhp_scale,
    bhp_rel_floor = SPE1_RATE_REL_FLOOR,
    bhp_weight = BHP_WEIGHT,
)
set_rate_observations!(registry, rate_obs)
history_matching_loss = loss_from_registry(registry; mode = LOSS_MODE)

# ---------------- Gradient setup ----------------

# layered initial guess (mD)
K_first_layer_guess = 750.0
K_second_layer_guess = 200.0
K_third_layer_guess = 1000.0
md_per_SI = 1000.0 / si_unit(:darcy)

sz = size(data["GRID"]["PERMX"])
K_guess = stack([
    fill(K_first_layer_guess / md_per_SI, sz[1:2]),
    fill(K_second_layer_guess / md_per_SI, sz[1:2]),
    fill(K_third_layer_guess / md_per_SI, sz[1:2]),
])

base_perm_vec = K_guess[:]
zonation = collect(1:length(base_perm_vec))
mults = ones(length(zonation))

function model(prm, step_info = missing)
    data_c = deepcopy(data)
    sz_local = size(data_c["GRID"]["PERMX"])
    permxyz = reshape(prm["perm"], sz_local)
    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = permxyz
    return setup_case_from_data_file(data_c)
end

println("Gradient computation (first call) @time:")
_ = @time begin
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            case_gradients(mults, zonation, K_guess, model, history_matching_loss)
        end
    end
end

println("Gradient computation (hot) @time:")
_ = @time begin
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            case_gradients(mults, zonation, K_guess, model, history_matching_loss)
        end
    end
end
