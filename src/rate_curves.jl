function wells_dict(well_results)
    raw = hasproperty(well_results, :wells) ? well_results.wells : well_results
    return Dict{Symbol, Any}(raw)
end

function production_wells(wells::Dict{Symbol, Any})
    prod = Symbol[]
    for (name, data) in pairs(wells)
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

function aggregate_rate_curve(wells::Dict{Symbol, Any},
                              qoi::Symbol;
                              well_names::Vector{Symbol})
    isempty(well_names) && error("Need at least one well to aggregate $qoi.")
    first_series = get(wells[well_names[1]], qoi) do
        error("Missing $qoi for well $(well_names[1])")
    end
    agg = zeros(Float64, length(first_series))
    agg .+= (sum(first_series) < 0 ? -1.0 : 1.0) .* first_series
    for w in Iterators.drop(well_names, 1)
        series = get(wells[w], qoi) do
            error("Missing $qoi for well $w")
        end
        length(series) == length(agg) || error("Mismatching length for $w/$qoi")
        agg .+= (sum(series) < 0 ? -1.0 : 1.0) .* series
    end
    return agg
end

aggregate_rates(wells::Dict{Symbol, Any},
                prod_wells::Vector{Symbol};
                qois = (:grat, :orat, :wrat)) =
    Dict(q => aggregate_rate_curve(wells, q; well_names = prod_wells) for q in qois)

function aggregate_bhp_curve(wells::Dict{Symbol, Any}; well_names::Vector{Symbol})
    isempty(well_names) && error("Need at least one well to aggregate bhp.")
    first_series = get(wells[well_names[1]], :bhp) do
        error("Missing bhp for well $(well_names[1])")
    end
    agg = zeros(Float64, length(first_series))
    for w in well_names
        series = get(wells[w], :bhp) do
            error("Missing bhp for well $w")
        end
        length(series) == length(agg) || error("Mismatching length for $w/bhp")
        agg .+= series
    end
    return agg ./ length(well_names)
end

function collect_rate_curves(wells_truth::Dict{Symbol, Any},
                             wells_matched::Dict{Symbol, Any};
                             qois = (:grat, :orat, :wrat),
                             prod_wells = nothing)
    prod = prod_wells === nothing ? production_wells(wells_truth) : prod_wells
    isempty(prod) && (prod = collect(keys(wells_truth)))
    curves = Dict{Symbol, NamedTuple{(:truth, :matched), Tuple{Vector{Float64}, Vector{Float64}}}}()
    for qoi in qois
        truth_curve = aggregate_rate_curve(wells_truth, qoi; well_names = prod)
        matched_curve = aggregate_rate_curve(wells_matched, qoi; well_names = prod)
        curves[qoi] = (truth = truth_curve, matched = matched_curve)
    end
    return curves, prod
end

function load_zonation_rate_curves(path::AbstractString)
    isfile(path) || return nothing
    qois = (:grat, :orat, :wrat)
    truth_vals = Dict(q => Float64[] for q in qois)
    matched_vals = Dict(q => Float64[] for q in qois)
    time_days = Float64[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(strip(line), ",")
            length(fields) < 1 + 2 * length(qois) && continue
            push!(time_days, parse(Float64, fields[1]))
            idx = 2
            for q in qois
                push!(truth_vals[q], parse(Float64, fields[idx]))
                push!(matched_vals[q], parse(Float64, fields[idx + 1]))
                idx += 2
            end
        end
    end
    curves = Dict(q => (truth = truth_vals[q], matched = matched_vals[q]) for q in qois)
    return (time_days = time_days, curves = curves)
end
