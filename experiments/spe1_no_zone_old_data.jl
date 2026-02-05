import Pkg
Pkg.activate(".")
Pkg.instantiate()

# ## Load packages
using Jutul, JutulDarcy, Printf, MultipliersZonation
using GeoEnergyIO
using GLMakie

# --- 3D visualization ---
GLMakie.activate!()

# --- palette and symmetric scale ---
const COLORMAP_GRAD  = :balance
const COLORMAP_MULTS = :viridis

const SPE1_WELLS          = [:PROD, :INJ]
const SPE1_RATE_SCALE     = 1.0
const SPE1_RATE_REL_FLOOR = 1e-2
const SPE1_RATE_WEIGHT    = 0.1
const SPE1_BHP_SCALE      = si_unit(:bar)
const SPE1_BHP_WEIGHT     = 10.0

# Multipliers bounds (relative)
const MIN_MULTIPLIER = 5e-2
const MAX_MULTIPLIER = 5.0

const REFINEMENT_NAME = "L-BFGS per-cell perm (no zonation)"
const SPE1_ZONE_RATE_CURVES_FILE = joinpath(@__DIR__, "logs", "spe1_zone_rate_curves.csv")

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

K_first_layer  = 500.0
K_second_layer = 50.0
K_third_layer  = 200.0

K_first_layer_guess  = 750.0
K_second_layer_guess = 200.0
K_third_layer_guess  = 1000.0

rate_scales, bhp_scale = _auto_scales(ws, SPE1_WELLS; method = :rms)

LOSS_REGISTRY = default_loss_registry()

rate_obs = build_rate_observations(
    ws;
    wells          = SPE1_WELLS,
    step_times     = step_times,
    total_time     = total_time,
    rate_scale     = SPE1_RATE_SCALE,
    rate_scales    = rate_scales,
    rate_rel_floor = SPE1_RATE_REL_FLOOR,
    rate_weight    = SPE1_RATE_WEIGHT,
    bhp_scale      = SPE1_BHP_SCALE,
    bhp_weight     = SPE1_BHP_WEIGHT,
)
set_rate_observations!(LOSS_REGISTRY, rate_obs)

history_matching_loss(m, s, dt, step_info, forces) =
    rates_mismatch(LOSS_REGISTRY, m, s, dt, step_info, forces)

# Original model: takes permeability directly
function model(prm, step_info = missing)
    data_c = deepcopy(data)
    sz = size(data_c["GRID"]["PERMX"])

    permxyz = reshape(prm["perm"], sz)

    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = data["GRID"]["PERMZ"]

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
    data_c["GRID"]["PERMZ"] = data["GRID"]["PERMZ"]

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
    rel_min = MIN_MULTIPLIER,
    rel_max = MAX_MULTIPLIER,
)

mults_tuned = optimize_reservoir(
    dopt,
    history_matching_loss;
    max_it             = 30,
    step_init          = 1e-2,
    max_initial_update = 5e-2,
)

log_lbfgs_history(
    dopt;
    grad_tol           = LBFGS_DEFAULTS.grad_tol,
    obj_change_tol     = LBFGS_DEFAULTS.obj_change_tol,
    obj_change_tol_rel = LBFGS_DEFAULTS.obj_change_tol_rel,
    max_it           = 30,
)

# Optimised multipliers and resulting permeability
mults = mults_tuned["mults"]

perm_opt_SI    = base_perm_vec .* mults[zonation]
final_perm_SI  = reshape(perm_opt_SI, sz)
final_perm_mD  = final_perm_SI .* md_per_SI

no_zone_perm_inc_file = joinpath(@__DIR__, "..", "models", "inc", "spe1_no_zone_perm.inc")
write_permeability_inc(
    no_zone_perm_inc_file;
    permx = final_perm_mD,
    permy = final_perm_mD,
    header_lines = [
        "-- SPE1 permeability tuned with per-cell L-BFGS (no zonation)",
        "-- Units: milliDarcy, order = I fastest, then J, then K",
    ],
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

show_epoch_gradients(rmesh, history.grads[last_epoch], history.labels[last_epoch]; colormap = COLORMAP_GRAD)
show_multipliers_line(history.mults[last_epoch]; title = history.labels[last_epoch])
show_multipliers_3d(
    rmesh,
    multiplier_field_for_refinement(history, last_epoch);
    colormap = COLORMAP_MULTS,
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
    colormap = COLORMAP_MULTS,
    title = @sprintf("Permeability field (mD) — %s", history.labels[last_epoch]),
)
show_perm_3d(
    rmesh,
    final_perm_mD;
    units = :mD,
    colormap = COLORMAP_MULTS,
    title = "Permeability field (mD) — L-BFGS per-cell perm",
)

# quick one-off log-scale plot of LBFGS loss per iteration
lbfgs_iter_indices = 0:10
lbfgs_iter_losses = [
    604255.7605190848,
    68907.4491452643,
    66473.41922833382,
    62895.417107781126,
    60042.950430383404,
    59930.01938879271,
    59929.48402793784,
    59928.788021837994,
    59928.773701765545,
    59928.77012084912,
    59928.77012084912,
]

fig_lbfgs = Figure()
ax_lbfgs = Axis(fig_lbfgs[1, 1];
    xlabel = "LBFGS iteration",
    ylabel = "Loss",
    title  = "Loss per iteration — L-BFGS (no zonation)",
    yscale = Makie.log10,
)
lines!(ax_lbfgs, lbfgs_iter_indices, lbfgs_iter_losses; linewidth = 2)
scatter!(ax_lbfgs, lbfgs_iter_indices, lbfgs_iter_losses; markersize = 10)
display(fig_lbfgs)

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

#TODO
#The unnececary plotting of rates < saivng of rates, and loging of loss function, make it more consistence in term of 
#moving it into functions and modules in another file in src and make it callable from here 
#and flexible for use on every model
