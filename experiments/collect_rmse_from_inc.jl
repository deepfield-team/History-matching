import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Jutul, JutulDarcy, MultipliersZonation, Statistics, GeoEnergyIO, Printf

# ----- Constants (no env vars) -----
const SPE1_WELLS = [:PROD, :INJ]
const INC_DIR = joinpath(@__DIR__, "..", "models", "inc")
const ZONING_TECHNIQUES = [
    "sign_uncons",
    "sign_cons",
    "medium-cons",
    "medium_uncons",
    "hierarchical clustering",
]
const NO_ZONE_LABEL = "no_zonation"
const NO_ZONE_FILE = joinpath(INC_DIR, "spe1_no_zone_perm.inc")
const OUTPUT_MD = joinpath(@__DIR__, "zonation_rmse_summary.md")
const BHP_TO_BAR = 1e5

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

function load_spe1_base()
    data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
    data = parse_data_file(data_pth)
    case_truth = setup_case_from_data_file(data)
    result_truth = simulate_reservoir(case_truth)
    ws = result_truth.wells
    step_times = cumsum(case_truth.dt)
    total_time = step_times[end]
    md_per_SI = 1000.0 / si_unit(:darcy)
    sz = size(data["GRID"]["PERMX"])
    return (; data, case_truth, result_truth, ws, step_times, total_time, md_per_SI, sz)
end

function read_perm_inc(path::AbstractString)
    isfile(path) || error("Include file not found at $path")
    vals = Float64[]
    in_permx = false
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "--") && continue
        if !in_permx
            s == "PERMX" && (in_permx = true)
            continue
        end
        s == "/" && break
        for tok in split(s)
            push!(vals, parse(Float64, tok))
        end
    end
    in_permx || error("PERMX block not found in $path")
    return vals
end

function simulate_with_perm(base, perm_mD)
    perm_SI = perm_mD ./ base.md_per_SI
    data_c = deepcopy(base.data)
    data_c["GRID"]["PERMX"] = perm_SI
    data_c["GRID"]["PERMY"] = perm_SI
    data_c["GRID"]["PERMZ"] = perm_SI
    case = setup_case_from_data_file(data_c)
    return simulate_reservoir(case)
end

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

    bhp_truth_prod = aggregate_bhp_curve(wells_truth; well_names = prod_wells) ./ BHP_TO_BAR
    bhp_matched_prod = aggregate_bhp_curve(wells_matched; well_names = prod_wells) ./ BHP_TO_BAR
    bhp_truth_inj = aggregate_bhp_curve(wells_truth; well_names = inj_wells) ./ BHP_TO_BAR
    bhp_matched_inj = aggregate_bhp_curve(wells_matched; well_names = inj_wells) ./ BHP_TO_BAR

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

function write_md(path::AbstractString, results::Vector{ZoneRunResult})
    header = [
        "Technique",
        "RMSE ORAT (prod)",
        "RMSE GRAT (prod)",
        "RMSE WRAT (prod)",
        "RMSE BHP (prod)",
        "RMSE BHP (inj)",
        "Perm RMSE L1 (mD)",
        "Perm RMSE L2 (mD)",
        "Perm RMSE L3 (mD)",
    ]
    fmt(x) = @sprintf("%.4g", x)

    open(path, "w") do io
        println(io, "# SPE1 RMSE Summary")
        println(io)
        println(io, "Rates are aggregated over production wells; BHP is averaged across prod/inj wells.")
        println(io, "BHP RMSE is reported in bar.")
        println(io)
        println(io, "| " * join(header, " | ") * " |")
        println(io, "|" * join(fill("---", length(header)), "|") * "|")
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
            println(io, "| " * join(row, " | ") * " |")
        end
    end
end

function main()
    base = load_spe1_base()

    results = ZoneRunResult[]

    for label in ZONING_TECHNIQUES
        inc_path = joinpath(INC_DIR, @sprintf("spe1_zone_LBFGS_perm_%s.inc", label))
        vals = read_perm_inc(inc_path)
        length(vals) == prod(base.sz) || error("PERMX length mismatch for $label")
        perm_mD = reshape(vals, base.sz)
        result_matched = simulate_with_perm(base, perm_mD)
        metrics = compute_metrics(base, result_matched; final_perm_mD = perm_mD)
        push!(results, ZoneRunResult(label, metrics.rmse_orat, metrics.rmse_grat, metrics.rmse_wrat,
            metrics.rmse_bhp_prod, metrics.rmse_bhp_inj, metrics.perm_rmse_layer1, metrics.perm_rmse_layer2, metrics.perm_rmse_layer3))
    end

    vals = read_perm_inc(NO_ZONE_FILE)
    length(vals) == prod(base.sz) || error("PERMX length mismatch for no-zonation")
    perm_mD = reshape(vals, base.sz)
    result_matched = simulate_with_perm(base, perm_mD)
    metrics = compute_metrics(base, result_matched; final_perm_mD = perm_mD)
    push!(results, ZoneRunResult(NO_ZONE_LABEL, metrics.rmse_orat, metrics.rmse_grat, metrics.rmse_wrat,
        metrics.rmse_bhp_prod, metrics.rmse_bhp_inj, metrics.perm_rmse_layer1, metrics.perm_rmse_layer2, metrics.perm_rmse_layer3))

    write_md(OUTPUT_MD, results)
    println("Wrote RMSE summary to $(OUTPUT_MD)")
end

main()
