import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Jutul, JutulDarcy, Printf, MultipliersZonation, Statistics
using GeoEnergyIO
using GLMakie
GLMakie.activate!()

# --- Controls ---
const COLORMAP_GRAD  = :balance
const COLORMAP_MULTS = :viridis
const COLORMAP_MULTS_3D = cgrad(:Spectral, 256, rev = true)

const SPE1_WELLS = [:PROD, :INJ]
const loss_mode = :rates
const scale_mode = :auto_rms
const bhp_scale_mode = :auto

const SPE1_RATE_REL_FLOOR = 1e-2
const rate_weight = 1.0
const bhp_weight = rate_weight

const MIN_MULTIPLIER = 5e-2
const MAX_MULTIPLIER = 1.5
const N_REFINEMENTS = 3
const LBFGS_MAX_ITERS = 40

const ZONING_TECHNIQUE_KEY = "sign_uncons"
const REFINEMENT_KEY = ZONING_TECHNIQUE_KEY
const GRADIENT_ZERO_TOL = 1e-12

const K_first_layer  = 500.0
const K_second_layer = 50.0
const K_third_layer  = 200.0

const K_first_layer_guess  = 750.0
const K_second_layer_guess = 200.0
const K_third_layer_guess  = 1000.0

const LAYER_LABELS = ("first", "second", "third")

const LBFGS_LOG_DIR  = joinpath(@__DIR__, "logs")
const SPE1_LOG_TAG = "$(scale_mode)_$(loss_mode)_$(bhp_scale_mode)"
const LBFGS_LOG_FILE = joinpath(LBFGS_LOG_DIR, "spe1_zone_LBFGS_lbfgs_$(SPE1_LOG_TAG)_$(REFINEMENT_KEY).csv")
const LBFGS_LOG_HEADER = "L-BFGS iteration log for spe1_zone_LBFGS.jl"
const RATE_CURVES_FILE = joinpath(LBFGS_LOG_DIR, "spe1_zone_rate_curves_$(SPE1_LOG_TAG)_$(REFINEMENT_KEY).csv")
const ZONE_PERM_INC_FILE = joinpath(@__DIR__, "..", "models", "inc", "spe1_zone_LBFGS_perm_$(REFINEMENT_KEY).inc")
const PERM_INC_HEADER_LINES = [
    "-- SPE1 permeability tuned with zonation-aware L-BFGS",
    "-- Units: milliDarcy, order = I fastest, then J, then K",
]

const REF_IDX_TO_PLOT = 7
const LBFGS_FIRST_REFINEMENT_TO_PLOT = 1
const LBFGS_LAST_REFINEMENT_TO_PLOT  = 10

const MULTS_LINE_YLABEL = "Permeability multiplier"
const MULTS_COLORBAR_LABEL = "Permeability multiplier"
const MULTS_COLORRANGE = (0.02, 1.5)
const PERM_COLORRANGE = (50.0, 1000.0)
const RATE_COMPARE_LABEL = "matched"

const LABEL_ZONES_TEMPLATE = "%s (%d zones)"
const TITLE_MULTS_PER_ZONE_TEMPLATE = "Multipliers per zone — %s (%d zones)"
const TITLE_MULTS_PER_CELL_TEMPLATE = "Multipliers per cell — %s (%d zones)"
const TITLE_PERM_TEMPLATE = "Permeability field (mD) - %s"
const TITLE_PERM_TRUTH = "Permeability field (mD) - truth"
const TITLE_PERM_GUESS = "Permeability field (mD) - initial guess"

const ZONING_TECHNIQUES = Dict(
    "sign_uncons" => (
        name = "sign_uncons",
        fn = (mults, zonation, rmesh, K_grads, mults_grads, refinement_step) ->
            MultipliersZonation.gradient_sign_refine!(
                mults, zonation, K_grads; zero_tolerance = GRADIENT_ZERO_TOL
            ),
    ),
    "sign_cons" => (
        name = "sign_cons",
        fn = (mults, zonation, rmesh, K_grads, mults_grads, refinement_step) ->
            MultipliersZonation.gradient_sign_targeted_refine!(
                mults, zonation, K_grads, GRADIENT_ZERO_TOL
            ),
    ),
    "medium-cons" => (
        name = "medium-cons",
        fn = (mults, zonation, rmesh, K_grads, mults_grads, refinement_step) ->
            MultipliersZonation.gradient_median_descending_refine!(
                mults, zonation, K_grads
            ),
    ),
    "medium_uncons" => (
        name = "medium_uncons",
        fn = (mults, zonation, rmesh, K_grads, mults_grads, refinement_step) ->
            MultipliersZonation.gradient_median_refine_all!(
                mults, zonation, K_grads
            ),
    ),
    "hierarchical clustering" => (
        name = "hierarchical clustering",
        fn = (mults, zonation, rmesh, K_grads, mults_grads, refinement_step) ->
            MultipliersZonation.incremental_gradient_quantile_refine!(
                mults,
                zonation,
                rmesh,
                K_grads,
                mults_grads,
                refinement_step;
                min_multiplier = MIN_MULTIPLIER,
                max_multiplier = MAX_MULTIPLIER,
            ),
    ),
)

# --- Load SPE1 model ---
data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
data = parse_data_file(data_pth)
case_truth = setup_case_from_data_file(data)
result = simulate_reservoir(case_truth)
ws     = result.wells
states = result.states

rmesh = physical_representation(reservoir_domain(case_truth.model))
step_times = cumsum(case_truth.dt)
total_time = step_times[end]

sz = size(data["GRID"]["PERMX"])
md_per_SI = 1000.0 / si_unit(:darcy)

K_first_layer_guess_SI  = fill(K_first_layer_guess  / md_per_SI, sz[1:2])
K_second_layer_guess_SI = fill(K_second_layer_guess / md_per_SI, sz[1:2])
K_third_layer_guess_SI  = fill(K_third_layer_guess  / md_per_SI, sz[1:2])

K_guess = stack([K_first_layer_guess_SI,
                 K_second_layer_guess_SI,
                 K_third_layer_guess_SI])

K_vector       = K_guess[:]
truth_perm_vec = vec(data["GRID"]["PERMX"])

# Backward-compatible wrapper: tolerate older compute_auto_scales signatures
_auto_scales(ws, wells; method = :rms) = try
    compute_auto_scales(ws, wells; method = method)
catch err
    err isa MethodError || rethrow()
    compute_auto_scales(ws, wells)
end

_fmt(template, args...) = Printf.format(Printf.Format(template), args...)


# --- Loss configuration ---
_resolve_refinement(ref, last_refinement) = begin
    ref isa Integer && return clamp(ref, 1, last_refinement)
    error("Invalid refinement value '$ref'. Use an integer index.")
end

rate_scales, bhp_scale = _auto_scales(ws, SPE1_WELLS; method = :rms)

LOSS_REGISTRY = default_loss_registry()

rate_obs = build_rate_observations(
    ws;
    wells          = SPE1_WELLS,
    step_times     = step_times,
    total_time     = total_time,
    rate_scales    = rate_scales,
    rate_rel_floor = SPE1_RATE_REL_FLOOR,
    rate_weight    = rate_weight,
    bhp_scale      = bhp_scale,
    bhp_rel_floor  = SPE1_RATE_REL_FLOOR,
    bhp_weight     = bhp_weight,
)
set_rate_observations!(LOSS_REGISTRY, rate_obs)
history_matching_loss = loss_from_registry(LOSS_REGISTRY; mode = loss_mode)

# --- Zonation configuration ---

zonation_config = get(ZONING_TECHNIQUES, REFINEMENT_KEY, nothing)
zonation_config === nothing && error("Unknown zoning technique key: $(REFINEMENT_KEY)")
refinement_name = zonation_config.name
refine_strategy = zonation_config.fn

n_refinements = N_REFINEMENTS
mults    = [1.0]
zonation = fill(1, number_of_cells(rmesh))
zonation_ref = Ref(zonation)

function model(prm, step_info = missing)
    data_c = deepcopy(data)
    sz = size(data_c["GRID"]["PERMX"])
    permxyz = reshape(prm["perm"], sz)
    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = permxyz
    return setup_case_from_data_file(data_c)
end

function model_mults(prm, step_info = missing)
    data_c = deepcopy(data)
    base_perm_vec = K_guess[:]
    perm_vec = base_perm_vec .* prm["mults"][zonation_ref[]]
    sz = size(data_c["GRID"]["PERMX"])
    permxyz = reshape(perm_vec, sz)
    data_c["GRID"]["PERMX"] = permxyz
    data_c["GRID"]["PERMY"] = permxyz
    data_c["GRID"]["PERMZ"] = permxyz
    return setup_case_from_data_file(data_c)
end

history = RefinementHistory()
reset_lbfgs_log_file!(LBFGS_LOG_FILE; header = LBFGS_LOG_HEADER)

gradient_fn = function (curr_mults, curr_zonation)
    loss, K_grads, mults_grads = redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            case_gradients(curr_mults, curr_zonation, K_guess, model, history_matching_loss)
        end
    end
    return loss, K_grads, mults_grads
end

optimizer_fn = function (curr_mults, curr_zonation, label, refinement_step)
    zonation_ref[] = curr_zonation
    optimize_multipliers!(
        curr_mults,
        model_mults,
        history_matching_loss;
        min_multiplier = MIN_MULTIPLIER,
        max_multiplier = MAX_MULTIPLIER,
        max_iters = LBFGS_MAX_ITERS,
        label = label,
        log_path = LBFGS_LOG_FILE,
        lbfgs_defaults = LBFGS_DEFAULTS,
    )
end

refinement_fn = (curr_mults, curr_zonation, K_grads, mults_grads, refinement_step) ->
    refine_strategy(
        curr_mults,
        curr_zonation,
        rmesh,
        K_grads,
        mults_grads,
        refinement_step,
    )

history, zonation = run_refinement_loop!(
    mults,
    zonation,
    refinement_fn,
    gradient_fn,
    optimizer_fn;
    n_refinements = n_refinements,
    refinement_name = refinement_name,
    history = history,
)
zonation_ref[] = zonation

last_refinement = history_length(history)

final_perm_SI = reshape(K_guess[:] .* mults[zonation], sz)
final_perm_mD = final_perm_SI .* md_per_SI

write_permeability_inc(
    ZONE_PERM_INC_FILE;
    permx = final_perm_mD,
    permy = final_perm_mD,
    header_lines = PERM_INC_HEADER_LINES,
)

prm_final = Dict("mults" => mults)
case_matched = model_mults(prm_final)
result_matched = simulate_reservoir(case_matched)
well_results_truth = ws
well_results_matched = result_matched.wells
time_truth = result.time
time_matched = result_matched.time

layer_truth  = [K_first_layer, K_second_layer, K_third_layer]
layer_labels = LAYER_LABELS
layer_means  = [mean(final_perm_mD[:, :, layer]) for layer in 1:sz[3]]

for (idx, label) in enumerate(layer_labels)
    tuned = layer_means[idx]
    truth = layer_truth[idx]
    err = abs(tuned - truth)
end

wells_truth_dict = wells_dict(well_results_truth)
wells_matched_dict = wells_dict(well_results_matched)
curves, _ = collect_rate_curves(wells_truth_dict, wells_matched_dict)
save_rate_curves!(RATE_CURVES_FILE, time_truth, curves)

perm_args = (history = history, base_perm_vec = K_vector, truth_perm_vec = truth_perm_vec, md_per_SI = md_per_SI)
ref_idx = _resolve_refinement(REF_IDX_TO_PLOT, last_refinement)
ref_label = history.labels[ref_idx]
ref_nzones = length(history.mults[ref_idx])

plot_epoch_gradients!(mesh, grads, label) = show_epoch_gradients(mesh, grads, label; colormap = COLORMAP_GRAD)
plot_multipliers_line!(mults, title) = show_multipliers_line(mults; title = title, ylabel = MULTS_LINE_YLABEL)
plot_multipliers_3d!(mesh, field, title) = show_multipliers_3d(mesh, field; colormap = COLORMAP_MULTS_3D, title = title, colorbar_label = MULTS_COLORBAR_LABEL, colorrange = MULTS_COLORRANGE)
plot_perm_3d!(mesh, field, title) = show_perm_3d(mesh, field; units = :mD, colormap = COLORMAP_MULTS, title = title, colorrange = PERM_COLORRANGE)
plot_loss_history!(losses, labels) = show_loss_history(losses, labels)
plot_rate_comparison!(t_truth, t_matched, curves) = show_rate_comparison(t_truth, t_matched, curves; label = RATE_COMPARE_LABEL)
plot_lbfgs_sections!(log_file, history) = begin
    lbfgs_sections = load_lbfgs_sections(log_file)
    isempty(lbfgs_sections.labels) && return @warn "No LBFGS sections found in $(log_file) — run optimization first."
    any(!isempty, lbfgs_sections.iters) ||
        return @warn "LBFGS sections in $(log_file) contain no iteration data — check logging output."
    lbfgs_labels = map(lbfgs_sections.labels) do lbl
        m = match(r"refinement\s+(\d+)", lbl)
        m === nothing && return "LBFGS loss"
        ref = parse(Int, m.captures[1])
        zones = ref ≤ history_length(history) ? length(history.mults[ref]) : missing
        zones isa Missing ? @sprintf("LBFGS loss — ref %d", ref) : @sprintf("LBFGS loss — ref %d (%d zones)", ref, zones)
    end
    nonempty = findall(!isempty, lbfgs_sections.iters)
    lbfgs_sections_clean = (
        labels = lbfgs_labels[nonempty],
        iters = lbfgs_sections.iters[nonempty],
        losses = lbfgs_sections.losses[nonempty],
    )
    show_lbfgs_sections(lbfgs_sections_clean; first_refinement = LBFGS_FIRST_REFINEMENT_TO_PLOT, last_refinement = LBFGS_LAST_REFINEMENT_TO_PLOT)
end

plot_epoch_gradients!(rmesh, history.grads[ref_idx], _fmt(LABEL_ZONES_TEMPLATE, ref_label, ref_nzones))
plot_multipliers_line!(history.mults[ref_idx], _fmt(TITLE_MULTS_PER_ZONE_TEMPLATE, ref_label, ref_nzones))
plot_multipliers_3d!(rmesh, multiplier_field_for_refinement(history, ref_idx), _fmt(TITLE_MULTS_PER_CELL_TEMPLATE, ref_label, ref_nzones))
plot_perm_3d!(rmesh, perm_field_for_refinement(ref_idx; perm_args...), _fmt(TITLE_PERM_TEMPLATE, ref_label))
plot_loss_history!(history.losses, history.labels)
plot_perm_3d!(rmesh, perm_field(:truth; perm_args...), TITLE_PERM_TRUTH)
plot_perm_3d!(rmesh, perm_field(:guess; perm_args...), TITLE_PERM_GUESS)
plot_perm_3d!(rmesh, perm_field(:final; perm_args...), _fmt(TITLE_PERM_TEMPLATE, history.labels[end]))
plot_lbfgs_sections!(LBFGS_LOG_FILE, history)
plot_rate_comparison!(time_truth, time_matched, curves)
