import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Jutul, JutulDarcy, MultipliersZonation, Statistics, GeoEnergyIO, Printf

# ----- Constants (no env vars) -----
const SPE1_WELLS = [:PROD, :INJ]
const LOSS_MODE = :rates
const SCALE_MODE = :auto_rms
const BHP_SCALE_MODE = :auto
const SPE1_RATE_REL_FLOOR = 1e-2
const RATE_WEIGHT = 1.0
const BHP_WEIGHT = RATE_WEIGHT
const MIN_MULTIPLIER = 5e-2
const MAX_MULTIPLIER = 1.5
const N_REFINEMENTS = 10
const LBFGS_MAX_ITERS = 40
const ZERO_TOL = 1e-12

struct ZoneRunResult
    label::String
    rmse_orat::Float64
    rmse_grat::Float64
    rmse_wrat::Float64
    rmse_bhp_prod::Float64
    rmse_bhp_inj::Float64
    perm_rmse_layer1::Float64
    perm_rmse_layer2::Float64
    perm_rmse_layer3::Float64
end

# ---------------- Utility helpers ----------------

wells_dict(well_results) = hasproperty(well_results, :wells) ? well_results.wells : well_results

function production_wells(wells::AbstractDict{Symbol, <:Any})
    prod = Symbol[]
    for (name, data) in wells
        total = 0.0
        for key in (:orat, :wrat, :grat)
            series = get(data, key, nothing)
            series === nothing && continue
            total += sum(series)
        end
        total < 0 && push!(prod, name)
    end
    return prod
end

function aggregate_rate_curve(wells::AbstractDict{Symbol, <:Any}, qoi::Symbol; well_names::Vector{Symbol})
    isempty(well_names) && error("Need wells to aggregate $qoi")
    first_series = get(wells[well_names[1]], qoi) do
        error("Missing $qoi for well $(well_names[1])")
    end
    agg = zeros(Float64, length(first_series))
    agg .+= (sum(first_series) < 0 ? -1.0 : 1.0) .* first_series
    for w in Iterators.drop(well_names, 1)
        series = get(wells[w], qoi) do
            error("Missing $qoi for well $w")
        end
        length(series) == length(agg) || error("Length mismatch for $w / $qoi")
        agg .+= (sum(series) < 0 ? -1.0 : 1.0) .* series
    end
    return agg
end

function aggregate_bhp_curve(wells::AbstractDict{Symbol, <:Any}; well_names::Vector{Symbol})
    isempty(well_names) && error("Need wells to aggregate bhp")
    base = wells[well_names[1]][:bhp]
    agg = zeros(Float64, length(base))
    for w in well_names
        series = get(wells[w], :bhp) do
            error("Missing bhp for well $w")
        end
        length(series) == length(base) || error("Length mismatch for $w / bhp")
        agg .+= series
    end
    agg ./= length(well_names)
    return agg
end

rmse(a::AbstractVector, b::AbstractVector) = sqrt(mean((a .- b) .^ 2))

# ---------------- Base model setup ----------------

function load_spe1_base()
    data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
    data = parse_data_file(data_pth)
    case_truth = setup_case_from_data_file(data)
    result_truth = simulate_reservoir(case_truth)
    ws = result_truth.wells
    step_times = cumsum(case_truth.dt)
    total_time = step_times[end]
    rmesh = physical_representation(reservoir_domain(case_truth.model))

    # layered initial guess (mD)
    K_first_layer_guess  = 750.0
    K_second_layer_guess = 200.0
    K_third_layer_guess  = 1000.0
    md_per_SI = 1000.0 / si_unit(:darcy)
    sz = size(data["GRID"]["PERMX"])
    K_guess = stack([
        fill(K_first_layer_guess  / md_per_SI, sz[1:2]),
        fill(K_second_layer_guess / md_per_SI, sz[1:2]),
        fill(K_third_layer_guess  / md_per_SI, sz[1:2]),
    ])

    truth_perm_vec = vec(data["GRID"]["PERMX"])

    return (; data, case_truth, result_truth, ws, step_times, total_time, rmesh, K_guess, truth_perm_vec, md_per_SI)
end

function build_loss(ws, step_times, total_time)
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
    return loss_from_registry(registry; mode = LOSS_MODE)
end

# Backward-compatible wrapper for compute_auto_scales
_auto_scales(ws, wells; method = :rms) = try
    compute_auto_scales(ws, wells; method = method)
catch err
    err isa MethodError || rethrow()
    compute_auto_scales(ws, wells)
end

# ---------------- Zonation runner ----------------

function run_zonation_strategy(base, loss_fn; refinement_label::String, refinement_fn::Function)
    data = base.data
    K_guess = base.K_guess
    md_per_SI = base.md_per_SI
    sz = size(data["GRID"]["PERMX"])

    mults = [1.0]
    zonation = fill(1, number_of_cells(base.rmesh))
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
        permxyz = reshape(perm_vec, sz)
        data_c["GRID"]["PERMX"] = permxyz
        data_c["GRID"]["PERMY"] = permxyz
        data_c["GRID"]["PERMZ"] = permxyz
        return setup_case_from_data_file(data_c)
    end

    gradient_fn = function (curr_mults, curr_zonation)
        loss, K_grads, mults_grads = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                case_gradients(curr_mults, curr_zonation, K_guess, model, loss_fn)
            end
        end
        return loss, K_grads, mults_grads
    end

    optimizer_fn = function (curr_mults, curr_zonation, label, refinement_step)
        zonation_ref[] = curr_zonation
        optimize_multipliers!(
            curr_mults,
            model_mults,
            loss_fn;
            min_multiplier = MIN_MULTIPLIER,
            max_multiplier = MAX_MULTIPLIER,
            max_iters = LBFGS_MAX_ITERS,
            label = label,
            log_path = nothing,
        )
    end

    history, zonation_final = run_refinement_loop!(
        mults,
        zonation,
        refinement_fn,
        gradient_fn,
        optimizer_fn;
        n_refinements = N_REFINEMENTS,
        refinement_name = refinement_label,
    )
    zonation_ref[] = zonation_final

    final_perm_SI = reshape(K_guess[:] .* mults[zonation_final], sz)
    final_perm_mD = final_perm_SI .* md_per_SI

    prm_final = Dict("mults" => mults)
    case_matched = model_mults(prm_final)
    result_matched = simulate_reservoir(case_matched)

    metrics = compute_metrics(base, result_matched; final_perm_mD = final_perm_mD)
    return metrics
end

# ---------------- No-zonation runner ----------------

function run_no_zonation(base, loss_fn)
    data = base.data
    K_guess = base.K_guess
    md_per_SI = base.md_per_SI
    sz = size(data["GRID"]["PERMX"])
    base_perm_vec = K_guess[:]
    zonation = collect(1:length(base_perm_vec))

    function model_mults(prm, step_info = missing)
        data_c = deepcopy(data)
        mult_field = prm["mults"][zonation]
        perm_vec = base_perm_vec .* mult_field
        permxyz = reshape(perm_vec, sz)
        data_c["GRID"]["PERMX"] = permxyz
        data_c["GRID"]["PERMY"] = permxyz
        data_c["GRID"]["PERMZ"] = permxyz
        return setup_case_from_data_file(data_c)
    end

    mults = ones(length(zonation))
    optimize_multipliers!(
        mults,
        model_mults,
        loss_fn;
        min_multiplier = MIN_MULTIPLIER,
        max_multiplier = MAX_MULTIPLIER,
        max_iters = 150,
        label = "L-BFGS no zonation",
        log_path = nothing,
    )

    final_perm_SI = reshape(base_perm_vec .* mults[zonation], sz)
    final_perm_mD = final_perm_SI .* md_per_SI

    prm_final = Dict("mults" => mults)
    case_matched = model_mults(prm_final)
    result_matched = simulate_reservoir(case_matched)

    metrics = compute_metrics(base, result_matched; final_perm_mD = final_perm_mD)
    return metrics
end

# ---------------- Metrics ----------------

function compute_metrics(base, result_matched; final_perm_mD)
    wells_truth = wells_dict(base.ws)
    wells_matched = wells_dict(result_matched.wells)

    prod_wells = production_wells(wells_truth)
    inj_wells = [w for w in keys(wells_truth) if !(w in prod_wells)]

    curves = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()
    for qoi in (:orat, :grat, :wrat)
        truth_curve = aggregate_rate_curve(wells_truth, qoi; well_names = prod_wells)
        matched_curve = aggregate_rate_curve(wells_matched, qoi; well_names = prod_wells)
        curves[qoi] = (truth_curve, matched_curve)
    end

    bhp_truth_prod = aggregate_bhp_curve(wells_truth; well_names = prod_wells)
    bhp_matched_prod = aggregate_bhp_curve(wells_matched; well_names = prod_wells)
    bhp_truth_inj = aggregate_bhp_curve(wells_truth; well_names = inj_wells)
    bhp_matched_inj = aggregate_bhp_curve(wells_matched; well_names = inj_wells)

    perm_truth_mD = base.data["GRID"]["PERMX"] .* base.md_per_SI

    layer_rmse = Vector{Float64}(undef, 3)
    for layer in 1:3
        truth_layer = perm_truth_mD[:, :, layer]
        pred_layer = final_perm_mD[:, :, layer]
        layer_rmse[layer] = rmse(vec(pred_layer), vec(truth_layer))
    end

    return ZoneRunResult(
        "",
        rmse(curves[:orat][1], curves[:orat][2]),
        rmse(curves[:grat][1], curves[:grat][2]),
        rmse(curves[:wrat][1], curves[:wrat][2]),
        rmse(bhp_truth_prod, bhp_matched_prod),
        rmse(bhp_truth_inj, bhp_matched_inj),
        layer_rmse[1],
        layer_rmse[2],
        layer_rmse[3],
    )
end

# ---------------- Entry point ----------------

function main()
    base = load_spe1_base()
    loss_fn = build_loss(base.ws, base.step_times, base.total_time)

    strategies = [
        ("sign_cons", (m, z, Kg, Mg, step) ->
            gradient_sign_targeted_refine!(m, z, Kg; zero_tolerance = ZERO_TOL)),
    ]

    results = ZoneRunResult[]
    labels = String[]

    for (label, refine_fn) in strategies
        println("Running zonation strategy: $label")
        metrics = run_zonation_strategy(base, loss_fn; refinement_label = label, refinement_fn = refine_fn)
        push!(results, ZoneRunResult(label, metrics.rmse_orat, metrics.rmse_grat, metrics.rmse_wrat,
            metrics.rmse_bhp_prod, metrics.rmse_bhp_inj, metrics.perm_rmse_layer1, metrics.perm_rmse_layer2, metrics.perm_rmse_layer3))
        push!(labels, label)
    end

    println("Running no-zonation baseline")
    baseline = run_no_zonation(base, loss_fn)
    push!(results, ZoneRunResult("no_zonation", baseline.rmse_orat, baseline.rmse_grat, baseline.rmse_wrat,
        baseline.rmse_bhp_prod, baseline.rmse_bhp_inj, baseline.perm_rmse_layer1, baseline.perm_rmse_layer2, baseline.perm_rmse_layer3))

    show_table(results)
end

function show_table(results::Vector{ZoneRunResult})
    header = ["Technique", "RMSE ORAT (prod)", "RMSE GRAT (prod)", "RMSE WRAT (prod)",
              "RMSE BHP (prod)", "RMSE BHP (inj)",
              "Perm RMSE L1 (mD)", "Perm RMSE L2 (mD)", "Perm RMSE L3 (mD)"]
    widths = [maximum(length(h) for h in header)]
    fmt(x) = @sprintf("%.4g", x)
    println("| " * join(header, " | ") * " |")
    println("|" * join(fill("---", length(header)), "|") * "|")
    for res in results
        row = [
            res.label,
            fmt(res.rmse_orat),
            fmt(res.rmse_grat),
            fmt(res.rmse_wrat),
            fmt(res.rmse_bhp_prod),
            fmt(res.rmse_bhp_inj),
            fmt(res.perm_rmse_layer1),
            fmt(res.perm_rmse_layer2),
            fmt(res.perm_rmse_layer3),
        ]
        println("| " * join(row, " | ") * " |")
    end
end

main()
