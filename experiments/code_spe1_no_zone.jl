import Pkg
Pkg.activate(".")
Pkg.instantiate()

# ## Load packages
using Jutul, JutulDarcy, Printf, MultipliersZonation
using GeoEnergyIO
using GLMakie

# --- 3D visualization ---
GLMakie.activate!()

# --- Controls ---
const SPE1_WELLS = [:PROD, :INJ]

const K_first_layer  = 500.0
const K_second_layer = 50.0
const K_third_layer  = 200.0

const K_first_layer_guess  = 750.0
const K_second_layer_guess = 200.0
const K_third_layer_guess  = 1000.0

const REFINEMENT_NAME = "L-BFGS per-cell perm (no zonation)"
const SPE1_ZONE_RATE_CURVES_FILE = joinpath(@__DIR__, "logs", "spe1_zone_rate_curves.csv")
const NO_ZONE_PERM_INC_FILE = joinpath(@__DIR__, "..", "models", "inc", "spe1_no_zone_perm.inc")
const PERM_INC_HEADER_LINES = [
    "-- SPE1 permeability tuned with per-cell L-BFGS (no zonation)",
    "-- Units: milliDarcy, order = I fastest, then J, then K",
]

# Backward-compatible wrapper: tolerate older compute_auto_scales signatures
_auto_scales(ws, wells; method = :rms) = try
    compute_auto_scales(ws, wells; method = method)
catch err
    err isa MethodError || rethrow()
    compute_auto_scales(ws, wells)
end

# ## Load SPE1 model
data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data = parse_data_file(data_pth);
case_truth = setup_case_from_data_file(data);
result = simulate_reservoir(case_truth)
ws     = result.wells

rmesh = physical_representation(reservoir_domain(case_truth.model))
step_times = cumsum(case_truth.dt)
total_time = step_times[end]

rate_scales, bhp_scale = _auto_scales(ws, SPE1_WELLS; method = :rms)

LOSS_REGISTRY = default_loss_registry()

rate_obs = build_rate_observations(
    ws;
    wells          = SPE1_WELLS,
    step_times     = step_times,
    total_time     = total_time,
    rate_scales    = rate_scales,
    rate_rel_floor = 1e-2,
    rate_weight    = 1.0,
    bhp_scale      = bhp_scale,
    bhp_rel_floor  = 1e-2,
    bhp_weight     = 1.0,
)
set_rate_observations!(LOSS_REGISTRY, rate_obs)

history_matching_loss = loss_from_registry(LOSS_REGISTRY; mode = :rates)

# Original model: takes permeability directly
function model(prm, step_info = missing)
    data_c = deepcopy(data)
    sz = size(data_c["GRID"]["PERMX"])

    permxyz = reshape(prm["perm"], sz)

    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = permxyz

    case = setup_case_from_data_file(data_c)
    return case
end

# --- initial layered guess for permeability (3 layers) ---
sz = size(data["GRID"]["PERMX"])
md_per_SI = 1000.0 / si_unit(:darcy)

K_first_layer_guess_SI  = fill(K_first_layer_guess  / md_per_SI, sz[1:2])
K_second_layer_guess_SI = fill(K_second_layer_guess / md_per_SI, sz[1:2])
K_third_layer_guess_SI  = fill(K_third_layer_guess  / md_per_SI, sz[1:2])

K_guess = stack([K_first_layer_guess_SI,
                 K_second_layer_guess_SI,
                 K_third_layer_guess_SI])

truth_perm_vec   = vec(data["GRID"]["PERMX"])

# Base permeability vector and trivial per-cell zonation
base_perm_vec = K_guess[:]
ncells        = length(base_perm_vec)
zonation      = collect(1:ncells)

# Model that uses multipliers: perm = base_perm_vec .* mults[zonation]
function model_mults(prm, step_info = missing)
    data_c = deepcopy(data)
    sz = size(data_c["GRID"]["PERMX"])

    mult_field = prm["mults"][zonation]
    perm_vec   = base_perm_vec .* mult_field
    permxyz    = reshape(perm_vec, sz)

    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = permxyz

    case = setup_case_from_data_file(data_c)
    return case
end

# --- L-BFGS on multipliers instead of perm ---
# Start from all-ones multipliers => initial perm = K_guess
mults0_vec = ones(length(zonation))

prm0 = Dict("mults" => copy(mults0_vec))

dopt = setup_reservoir_dict_optimization(prm0, model_mults)

free_optimization_parameter!(
    dopt,
    "mults";
    rel_min = 5e-2,
    rel_max = 1.5,
)

mults_tuned = optimize_reservoir(
    dopt,
    history_matching_loss;
)

log_lbfgs_history(
    dopt;
    grad_tol           = LBFGS_DEFAULTS.grad_tol,
    obj_change_tol     = LBFGS_DEFAULTS.obj_change_tol,
    obj_change_tol_rel = LBFGS_DEFAULTS.obj_change_tol_rel,
)

# Optimised multipliers and resulting permeability
mults = mults_tuned["mults"]

perm_opt_SI    = base_perm_vec .* mults[zonation]
final_perm_SI  = reshape(perm_opt_SI, sz)
final_perm_mD  = final_perm_SI .* md_per_SI

write_permeability_inc(
    NO_ZONE_PERM_INC_FILE;
    permx = final_perm_mD,
    permy = final_perm_mD,
    header_lines = PERM_INC_HEADER_LINES,
)

prm_final = Dict("perm" => perm_opt_SI)
case_matched = model(prm_final)
result_matched = simulate_reservoir(case_matched)
well_results_truth = ws
well_results_matched = result_matched.wells
time_truth = result.time
time_matched = result_matched.time

# --- history for visualisation ---
history = RefinementHistory()

# per-cell multipliers: k_opt = K_guess .* mults
mults = perm_opt_SI ./ base_perm_vec

loss_final, K_grads, mults_grads = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        case_gradients(mults, zonation, K_guess, model, history_matching_loss)
    end
end

label_final = "L-BFGS final"
log_refinement!(history, K_grads, mults, zonation, label_final, loss_final)

# ------------------ VISUALIZATION ------------------

# ------------------ PLOTTING CALLS AT THE END ------------------

last_epoch = history_length(history)

show_epoch_gradients(rmesh, history.grads[last_epoch], history.labels[last_epoch]; colormap = :balance)
show_multipliers_line(history.mults[last_epoch]; title = history.labels[last_epoch])
show_multipliers_3d(
    rmesh,
    multiplier_field_for_refinement(history, last_epoch);
    colormap = :viridis,
)
show_loss_history(history.losses, history.labels; technique_name = REFINEMENT_NAME)
show_perm_3d(
    rmesh,
    perm_field_for_refinement(last_epoch;
        history = history,
        base_perm_vec = base_perm_vec,
        truth_perm_vec = truth_perm_vec,
        md_per_SI = md_per_SI,
        units = :mD,
    );
    units = :mD,
    colormap = :viridis,
    title = @sprintf("Permeability field (mD) — %s", history.labels[last_epoch]),
)
show_perm_3d(
    rmesh,
    final_perm_mD;
    units = :mD,
    colormap = :viridis,
    title = "Permeability field (mD) — L-BFGS per-cell perm",
)


wells_truth_dict = wells_dict(well_results_truth)
wells_matched_dict = wells_dict(well_results_matched)
curves_no_zone, prod_wells = collect_rate_curves(wells_truth_dict, wells_matched_dict)

show_rate_comparison(
    time_truth,
    time_matched,
    curves_no_zone;
    label = "SPE1 — L-BFGS (no zonation)",
)

zonation_data = load_zonation_rate_curves(SPE1_ZONE_RATE_CURVES_FILE)
if zonation_data === nothing
    @warn "Zonation rate data not found at $SPE1_ZONE_RATE_CURVES_FILE — run spe1_zone_LBFGS.jl first for combined plot."
else
    show_rate_comparison_with_zonation(
        time_truth,
        curves_no_zone,
        zonation_data;
        label_no_zone = "No zonation",
        label_zonation = "Zonation",
    )
end
