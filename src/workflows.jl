using Printf: @sprintf
using LinearAlgebra: norm

const LBFGS_DEFAULTS = (
    grad_tol = 1e-6,
    obj_change_tol = 1e-6,
    obj_change_tol_rel = -Inf,
)

mutable struct RefinementHistory
    grads::Vector{Vector{Float64}}
    mults::Vector{Vector{Float64}}
    zones::Vector{Vector{Int}}
    losses::Vector{Float64}
    labels::Vector{String}
end

RefinementHistory() = RefinementHistory(
    Vector{Vector{Float64}}(),
    Vector{Vector{Float64}}(),
    Vector{Vector{Int}}(),
    Float64[],
    String[],
)

history_length(hist::RefinementHistory) = length(hist.labels)

function log_refinement!(hist::RefinementHistory, K_grads, mults, zonation, label, loss)
    push!(hist.grads, copy(K_grads))
    push!(hist.mults, copy(mults))
    push!(hist.zones, copy(zonation))
    push!(hist.losses, loss)
    push!(hist.labels, String(label))
    return hist
end

function loss_from_registry(registry::LossFunctionRegistry; mode = :rates)
    mode_sym = Symbol(mode)
    mode_sym == :rates ||
        error("Unknown loss mode $mode; use :rates.")
    return (m, s, dt, step_info, forces) ->
        rates_mismatch(registry, m, s, dt, step_info, forces)
end

function reset_lbfgs_log_file!(path::AbstractString; header::AbstractString = "L-BFGS iteration log")
    mkpath(dirname(path))
    open(path, "w") do io
        isempty(header) || println(io, "# ", header)
        println(io, "label,iter,loss,grad_norm,step,ls_iter,ls_flag")
    end
    return path
end

function _append_lbfgs_log_section(
        log_path::AbstractString,
        label::AbstractString,
        rows::Vector{<:NamedTuple},
    )
    mkpath(dirname(log_path))
    open(log_path, "a") do io
        println(io, "# section,$label")
        for row in rows
            println(io, join((
                row.label,
                row.iter,
                row.loss,
                row.grad_norm,
                row.step,
                row.ls_iter,
                row.ls_flag,
            ), ","))
        end
    end
    return nothing
end

function _lbfgs_history(dopt)
    hist = getproperty(dopt, :history)
    if isnothing(hist)
        return nothing
    elseif hist isa NamedTuple && (:history in keys(hist))
        return getfield(hist, :history)
    else
        return hist
    end
end

_format_float(x::Real) = begin
    y = float(x)
    if isnan(y)
        return "NaN"
    elseif isinf(y)
        return y > 0 ? "Inf" : "-Inf"
    else
        return @sprintf("%.16e", y)
    end
end
_format_float(::Missing) = "missing"

_format_numeric_or_missing(x::Missing) = "missing"
_format_numeric_or_missing(x::Real) = _format_float(x)
_format_numeric_or_missing(x) = string(x)

_grad_norm_entry(x::Missing) = NaN
_grad_norm_entry(x::Number) = float(x)
_grad_norm_entry(x) = norm(x, Inf)

function _lbfgs_stop_reason(
        grad_norm,
        niters,
        Δloss,
        relΔ,
        max_it,
        grad_tol,
        obj_change_tol,
        obj_change_tol_rel,
    )
    if grad_norm < grad_tol
        return @sprintf(
            "projected gradient norm %.3e < grad_tol %.3e",
            grad_norm, grad_tol,
        )
    elseif niters >= max_it
        return @sprintf("reached max_it = %d", max_it)
    elseif !isnan(Δloss) && Δloss < obj_change_tol
        return @sprintf("|Δloss| %.3e < obj_change_tol %.3e", Δloss, obj_change_tol)
    elseif isfinite(obj_change_tol_rel) && !isnan(relΔ) && relΔ < obj_change_tol_rel
        return @sprintf("|Δloss|/|loss| %.3e < obj_change_tol_rel %.3e", relΔ, obj_change_tol_rel)
    else
        return "time limit exceeded or optimizer exited without meeting tolerance checks"
    end
end

function log_lbfgs_history(
        dopt;
        label::AbstractString = "L-BFGS multipliers",
        grad_tol = LBFGS_DEFAULTS.grad_tol,
        obj_change_tol = LBFGS_DEFAULTS.obj_change_tol,
        obj_change_tol_rel = LBFGS_DEFAULTS.obj_change_tol_rel,
        max_it::Integer = 60,
        log_path::Union{Nothing, AbstractString} = nothing,
    )
    log_path === nothing && return nothing

    hist = _lbfgs_history(dopt)
    if hist === nothing
        @warn "$label — LBFGS history is unavailable; run optimize_reservoir before logging."
        return nothing
    end

    solver_hist = hasproperty(hist, :solver_history) ? getproperty(hist, :solver_history) :
                  (hasproperty(hist, :history) ? getproperty(hist, :history) : hist)

    vals = if solver_hist !== nothing && hasproperty(solver_hist, :val)
        getproperty(solver_hist, :val)
    elseif hasproperty(hist, :objectives)
        getproperty(hist, :objectives)
    else
        nothing
    end
    vals === nothing && begin
        @warn "$label — LBFGS history does not contain objective values."
        return nothing
    end
    if isempty(vals)
        @warn "$label — LBFGS history does not contain any iterations."
        return nothing
    end

    grads = if solver_hist !== nothing && hasproperty(solver_hist, :pg)
        getproperty(solver_hist, :pg)
    elseif hasproperty(hist, :gradient_norms)
        getproperty(hist, :gradient_norms)
    else
        nothing
    end

    pg_norm = grads === nothing ? Float64[] : [_grad_norm_entry(pg) for pg in grads]
    steps   = solver_hist !== nothing && hasproperty(solver_hist, :alpha) ? solver_hist.alpha : Any[]
    lsits   = solver_hist !== nothing && hasproperty(solver_hist, :lsit) ? solver_hist.lsit : Any[]
    lsfl    = solver_hist !== nothing && hasproperty(solver_hist, :lsfl) ? solver_hist.lsfl : Any[]

    nrec = length(vals)
    rows = NamedTuple[]
    for idx in 1:nrec
        iter = idx - 1
        loss = vals[idx]
        grad_norm = idx ≤ length(pg_norm) ? pg_norm[idx] : NaN
        step = idx ≤ length(steps) ? steps[idx] : NaN
        ls_iter = idx ≤ length(lsits) ? lsits[idx] : missing
        flag = idx ≤ length(lsfl) ? lsfl[idx] : missing
        push!(rows, (
            label = label,
            iter = string(iter),
            loss = _format_float(loss),
            grad_norm = _format_float(grad_norm),
            step = _format_numeric_or_missing(step),
            ls_iter = _format_numeric_or_missing(ls_iter),
            ls_flag = _format_numeric_or_missing(flag),
        ))
    end

    niters = max(nrec - 1, 0)
    Δloss = nrec ≥ 2 ? abs(vals[end] - vals[end - 1]) : NaN
    denom = max(abs(vals[end]), eps())
    relΔ = isnan(Δloss) ? NaN : abs(Δloss / denom)

    last_grad_norm = isempty(pg_norm) ? NaN : pg_norm[min(length(pg_norm), nrec)]
    reason = _lbfgs_stop_reason(
        last_grad_norm,
        niters,
        Δloss,
        relΔ,
        max_it,
        grad_tol,
        obj_change_tol,
        obj_change_tol_rel,
    )

    summary_row = (
        label = label,
        iter = "summary",
        loss = _format_float(vals[end]),
        grad_norm = _format_float(last_grad_norm),
        step = _format_numeric_or_missing(Δloss),
        ls_iter = _format_numeric_or_missing(relΔ),
        ls_flag = replace(reason, ',' => ';'),
    )
    push!(rows, summary_row)
    _append_lbfgs_log_section(log_path, label, rows)

    return nothing
end

function optimize_multipliers!(
        mults::AbstractVector,
        model_fn::Function,
        loss_fn::Function;
        min_multiplier = 5e-2,
        max_multiplier = 5.0,
        max_iters::Int = 60,
        label::AbstractString = "L-BFGS multipliers",
        log_path::Union{Nothing, AbstractString} = nothing,
        lbfgs_defaults = LBFGS_DEFAULTS,
    )
    prm = Dict("mults" => copy(mults))
    dopt = setup_reservoir_dict_optimization(prm, model_fn)

    free_optimization_parameter!(
        dopt,
        "mults";
        rel_min = min_multiplier,
        rel_max = max_multiplier,
    )

    tuned_prm = optimize_reservoir(
        dopt,
        loss_fn;
        max_it = max_iters,
    )

    log_lbfgs_history(
        dopt;
        label = label,
        grad_tol = lbfgs_defaults.grad_tol,
        obj_change_tol = lbfgs_defaults.obj_change_tol,
        obj_change_tol_rel = lbfgs_defaults.obj_change_tol_rel,
        max_it = max_iters,
        log_path = log_path,
    )

    mults .= tuned_prm["mults"]
    return nothing
end

function run_refinement_loop!(
        mults::AbstractVector,
        zonation::Vector{Int},
        refinement_strategy::Function,
        gradient_fn::Function,
        optimizer_fn::Function;
        n_refinements::Integer,
        refinement_name::AbstractString,
        history::RefinementHistory = RefinementHistory(),
    )
    for refinement_step in 1:n_refinements
        lbfgs_label = @sprintf("%s — refinement %d", refinement_name, refinement_step)
        optimizer_fn(mults, zonation, lbfgs_label, refinement_step)

        loss, K_grads, mults_grads = gradient_fn(mults, zonation)
        label = @sprintf("ref %d", refinement_step)
        log_refinement!(history, K_grads, mults, zonation, label, loss)

        zonation = refinement_strategy(
            mults,
            zonation,
            K_grads,
            mults_grads,
            refinement_step,
        )
    end

    return history, zonation
end

function _compress_labels(z::AbstractVector{<:Integer})
    lab2idx = Dict{Int, Int}()
    k = 0
    comp = similar(z, Int)
    @inbounds for i in eachindex(z)
        lbl = z[i]
        idx = get!(lab2idx, lbl) do
            k += 1
            k
        end
        comp[i] = idx
    end
    return comp, k
end

function multiplier_field_for_refinement(history::RefinementHistory, i::Integer)
    1 ≤ i ≤ history_length(history) || throw(ArgumentError("invalid refinement step index"))
    z = history.zones[i]
    vals = history.mults[i]
    comp, k = _compress_labels(z)
    k == length(vals) || throw(ArgumentError("refinement step $i: #zones ($k) != length(mults) ($(length(vals)))"))
    fld = Vector{Float64}(undef, length(comp))
    @inbounds for j in eachindex(comp)
        fld[j] = vals[comp[j]]
    end
    return fld
end

function _perm_SI_vec(kind::Symbol;
                     refinement_step::Integer,
                     history::RefinementHistory,
                     base_perm_vec::AbstractVector,
                     truth_perm_vec::Union{Nothing, AbstractVector},
    )
    if kind === :truth
        truth_perm_vec !== nothing ||
            throw(ArgumentError("truth_perm_vec is required when kind = :truth"))
        return truth_perm_vec

    elseif kind === :guess
        return base_perm_vec

    elseif kind === :refinement
        1 ≤ refinement_step ≤ history_length(history) ||
            throw(ArgumentError("invalid refinement step $refinement_step"))

        z  = history.zones[refinement_step]
        ms = history.mults[refinement_step]

        comp, k = _compress_labels(z)
        k == length(ms) ||
            throw(ArgumentError("refinement step $refinement_step: #zones ($k) != length(mults) ($(length(ms)))"))

        mult_field = Vector{Float64}(undef, length(comp))
        @inbounds for j in eachindex(comp)
            mult_field[j] = ms[comp[j]]
        end

        return base_perm_vec .* mult_field

    elseif kind === :final
        return _perm_SI_vec(:refinement;
            refinement_step = history_length(history),
            history = history,
            base_perm_vec = base_perm_vec,
            truth_perm_vec = truth_perm_vec,
        )

    else
        error("Unknown permeability kind: $kind (use :truth, :guess, :refinement, or :final)")
    end
end

function perm_field(
        kind::Symbol;
        history::RefinementHistory,
        refinement_step::Integer = history_length(history),
        base_perm_vec::AbstractVector,
        truth_perm_vec::Union{Nothing, AbstractVector},
        md_per_SI::Real = 1000.0 / si_unit(:darcy),
        units::Symbol = :mD,
    )
    perm_SI_vec = _perm_SI_vec(
        kind;
        refinement_step = refinement_step,
        history = history,
        base_perm_vec = base_perm_vec,
        truth_perm_vec = truth_perm_vec,
    )

    if units === :SI
        return perm_SI_vec
    elseif units === :mD
        return perm_SI_vec .* md_per_SI
    else
        error("Unknown units: $units (use :SI or :mD)")
    end
end

function perm_field_for_refinement(
        i::Integer;
        history::RefinementHistory,
        base_perm_vec::AbstractVector,
        truth_perm_vec::Union{Nothing, AbstractVector},
        md_per_SI::Real = 1000.0 / si_unit(:darcy),
        units::Symbol = :mD,
    )
    return perm_field(:refinement;
        refinement_step = i,
        history = history,
        base_perm_vec = base_perm_vec,
        truth_perm_vec = truth_perm_vec,
        md_per_SI = md_per_SI,
        units = units,
    )
end
