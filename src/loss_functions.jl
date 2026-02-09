using Statistics: std

struct StateReferenceSeries{S<:AbstractVector}
    step_times::Vector{Float64}
    total_time::Float64
    states::S
end

const PressureReference = StateReferenceSeries

struct RateObservations
    wells::Vector{Symbol}
    step_times::Vector{Float64}
    total_time::Float64
    obs_oil::Dict{Symbol, Vector{Float64}}
    obs_wat::Dict{Symbol, Vector{Float64}}
    obs_gas::Dict{Symbol, Vector{Float64}}
    obs_bhp::Dict{Symbol, Vector{Float64}}
    rate_scales::NamedTuple{(:oil, :wat, :gas), NTuple{3, Float64}}
    rate_floor::Float64
    rate_weight::Float64
    bhp_scale::Float64
    bhp_floor::Float64
    bhp_weight::Float64
end

# Backwards compatibility for existing construction calls that still pass a single
# rate scale (with or without a separate BHP relative error floor).
RateObservations(
    wells::Vector{Symbol},
    step_times::Vector{Float64},
    total_time::Float64,
    obs_oil::Dict{Symbol, Vector{Float64}},
    obs_wat::Dict{Symbol, Vector{Float64}},
    obs_gas::Dict{Symbol, Vector{Float64}},
    obs_bhp::Dict{Symbol, Vector{Float64}},
    rate_scale::Float64,
    rate_floor::Float64,
    rate_weight::Float64,
    bhp_scale::Float64,
    bhp_floor::Float64,
    bhp_weight::Float64,
) = RateObservations(
    wells,
    step_times,
    total_time,
    obs_oil,
    obs_wat,
    obs_gas,
    obs_bhp,
    (oil = rate_scale, wat = rate_scale, gas = rate_scale),
    rate_floor,
    rate_weight,
    bhp_scale,
    bhp_floor,
    bhp_weight,
)

RateObservations(
    wells::Vector{Symbol},
    step_times::Vector{Float64},
    total_time::Float64,
    obs_oil::Dict{Symbol, Vector{Float64}},
    obs_wat::Dict{Symbol, Vector{Float64}},
    obs_gas::Dict{Symbol, Vector{Float64}},
    obs_bhp::Dict{Symbol, Vector{Float64}},
    rate_scale::Float64,
    rate_floor::Float64,
    rate_weight::Float64,
    bhp_scale::Float64,
    bhp_weight::Float64,
) = RateObservations(
    wells,
    step_times,
    total_time,
    obs_oil,
    obs_wat,
    obs_gas,
    obs_bhp,
    (oil = rate_scale, wat = rate_scale, gas = rate_scale),
    rate_floor,
    rate_weight,
    bhp_scale,
    rate_floor,
    bhp_weight,
)

mutable struct LossFunctionRegistry
    state_reference::Union{Nothing, StateReferenceSeries}
    rates::Union{Nothing, RateObservations}
end

LossFunctionRegistry() = LossFunctionRegistry(nothing, nothing)

default_loss_registry() = LossFunctionRegistry()

function set_state_reference!(registry::LossFunctionRegistry; step_times, total_time, states)
    registry.state_reference = StateReferenceSeries(
        Float64.(collect(step_times)),
        Float64(total_time),
        states,
    )
    return registry
end

function set_pressure_reference!(registry::LossFunctionRegistry; step_times, total_time, states)
    Base.depwarn(
        "set_pressure_reference! is deprecated, use set_state_reference! instead.",
        :set_pressure_reference!,
    )
    return set_state_reference!(
        registry;
        step_times = step_times,
        total_time = total_time,
        states = states,
    )
end

function _normalize_rate_scales(rate_scale::Real, rate_scales)
    if rate_scales === nothing
        return (oil = Float64(rate_scale), wat = Float64(rate_scale), gas = Float64(rate_scale))
    end
    scales_dict = rate_scales isa AbstractDict ? rate_scales : Dict(pairs(rate_scales))
    oil_scale = get(scales_dict, :oil, get(scales_dict, :orat, nothing))
    wat_scale = get(scales_dict, :wat, get(scales_dict, :water, get(scales_dict, :wrat, nothing)))
    gas_scale = get(scales_dict, :gas, get(scales_dict, :grat, nothing))

    oil_scale === nothing && error("Missing oil rate scale in rate_scales")
    wat_scale === nothing && error("Missing water rate scale in rate_scales")
    gas_scale === nothing && error("Missing gas rate scale in rate_scales")

    return (oil = Float64(oil_scale), wat = Float64(wat_scale), gas = Float64(gas_scale))
end

function compute_auto_scales(well_results, wells; method::Symbol = :rms)
    method == :rms ||
        error("Unknown rate scale method $(method); use :rms for auto-scaling")
    names = Symbol.(collect(wells))
    rate_vals = Dict(:oil => Float64[], :wat => Float64[], :gas => Float64[])
    bhp_vals = Float64[]
    for w in names
        data = get(well_results.wells, w) do
            error("Missing well $w in simulation results")
        end
        haskey(data, :orat) && append!(rate_vals[:oil], abs.(Float64.(data[:orat])))
        haskey(data, :wrat) && append!(rate_vals[:wat], abs.(Float64.(data[:wrat])))
        haskey(data, :grat) && append!(rate_vals[:gas], abs.(Float64.(data[:grat])))
        if haskey(data, :bhp)
            append!(bhp_vals, abs.(Float64.(data[:bhp])))
        end
    end
    rms(v) = isempty(v) ? 0.0 : sqrt(sum(x -> x^2, v) / length(v))
    stat(v) = begin
        if isempty(v)
            return 0.0
        end
        s = rms(v)
        return isfinite(s) ? s : 0.0
    end
    compute_scale(vals, default; stat_fn = stat) = begin
        s = stat_fn(vals)
        return s > 0 ? s : (isempty(vals) ? default : maximum(vals))
    end
    raw_scales = (
        oil = compute_scale(rate_vals[:oil], 1.0),
        wat = compute_scale(rate_vals[:wat], 1.0),
        gas = compute_scale(rate_vals[:gas], 1.0),
    )
    pos_scales = filter(>(0.0), values(raw_scales))
    max_scale = isempty(pos_scales) ? 1.0 : maximum(pos_scales)
    min_floor = max_scale > 0 ? max_scale * 1e-6 : 1.0
    rate_scales = (
        oil = max(raw_scales.oil, min_floor),
        wat = max(raw_scales.wat, min_floor),
        gas = max(raw_scales.gas, min_floor),
    )
    bhp_scale = compute_scale(bhp_vals, si_unit(:bar); stat_fn = rms)
    return rate_scales, bhp_scale
end

function build_rate_observations(
    well_results;
    wells,
    step_times,
    total_time,
    rate_scale = 1.0,
    rate_scales = nothing,
    rate_rel_floor = 1e-2,
    rate_weight = 1.0,
    bhp_scale = si_unit(:bar),
    bhp_rel_floor = rate_rel_floor,
    bhp_weight = 1.0,
)
    scales = _normalize_rate_scales(rate_scale, rate_scales)
    names = Symbol.(collect(wells))
    obs_oil = Dict{Symbol, Vector{Float64}}()
    obs_wat = Dict{Symbol, Vector{Float64}}()
    obs_gas = Dict{Symbol, Vector{Float64}}()
    obs_bhp = Dict{Symbol, Vector{Float64}}()
    get_curve_length(data) = begin
        for key in (:orat, :wrat, :grat, :bhp)
            if haskey(data, key)
                return length(data[key])
            end
        end
        error("Unable to determine time-series length for well; available keys: $(collect(keys(data)))")
    end

    zero_curve(len) = zeros(Float64, len)

    for w in names
        data = get(well_results.wells, w) do
            error("Missing well $w in simulation results")
        end
        len = get_curve_length(data)
        obs_oil[w] = Float64.(get(data, :orat) do
            zero_curve(len)
        end) ./ scales.oil
        obs_wat[w] = Float64.(get(data, :wrat) do
            zero_curve(len)
        end) ./ scales.wat
        obs_gas[w] = Float64.(get(data, :grat) do
            zero_curve(len)
        end) ./ scales.gas
        obs_bhp[w] = Float64.(get(data, :bhp) do
            zero_curve(len)
        end) ./ bhp_scale
    end
    return RateObservations(
        names,
        Float64.(collect(step_times)),
        Float64(total_time),
        obs_oil,
        obs_wat,
        obs_gas,
        obs_bhp,
        scales,
        rate_rel_floor,
        rate_weight,
        bhp_scale,
        bhp_rel_floor,
        bhp_weight,
    )
end

function set_rate_observations!(registry::LossFunctionRegistry, obs::RateObservations)
    registry.rates = obs
    return registry
end

pdiff(p, p0) = sum((p[i] - p0[i])^2 for i in eachindex(p))

@inline _step_index(times, t) = clamp(searchsortedlast(times, t), 1, length(times))

function reservoir_pressure_mismatch(registry::LossFunctionRegistry, model, state, dt, step_info, forces)
    pref = registry.state_reference
    pref === nothing && error("Reference states not configured")
    t = step_info[:time] + dt
    step = _step_index(pref.step_times, t)
    p = state[:Reservoir][:Pressure]
    ref = pref.states[step][:Pressure]
    return (dt / pref.total_time) * (pdiff(p, ref) / (si_unit(:bar) * 100)^2)
end

@inline rel_err(value, ref, floor) = (value - ref) / max(abs(ref), floor)

function rates_mismatch(registry::LossFunctionRegistry, model, state, dt, step_info, forces)
    obs = registry.rates
    obs === nothing && error("Rate observations not configured")
    t = step_info[:time] + dt
    step = _step_index(obs.step_times, t)

    s = 0.0
    for w in obs.wells
        qo = JutulDarcy.compute_well_qoi(model, state, forces, w, :orat) / obs.rate_scales.oil
        qw = JutulDarcy.compute_well_qoi(model, state, forces, w, :wrat) / obs.rate_scales.wat
        qg = JutulDarcy.compute_well_qoi(model, state, forces, w, :grat) / obs.rate_scales.gas
        qb = JutulDarcy.compute_well_qoi(model, state, forces, w, :bhp) / obs.bhp_scale

        err_o = qo - obs.obs_oil[w][step]
        err_w = qw - obs.obs_wat[w][step]
        err_g = qg - obs.obs_gas[w][step]
        err_b = qb - obs.obs_bhp[w][step]

        s += obs.rate_weight * (err_o^2 + err_w^2 + err_g^2) / 3
        s += obs.bhp_weight * err_b^2
    end

    return (dt / obs.total_time) * s / length(obs.wells)
end

function bhp_mismatch(registry::LossFunctionRegistry, model, state, dt, step_info, forces)
    obs = registry.rates
    obs === nothing && error("Rate observations not configured")
    t = step_info[:time] + dt
    step = _step_index(obs.step_times, t)

    s = 0.0
    for w in obs.wells
        qb = JutulDarcy.compute_well_qoi(model, state, forces, w, :bhp) / obs.bhp_scale
        err_b = qb - obs.obs_bhp[w][step]
        s += obs.bhp_weight * err_b^2
    end

    return (dt / obs.total_time) * s / length(obs.wells)
end
