import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Jutul
using GLMakie
using MultipliersZonation

GLMakie.activate!()

const TRAIN_YEARS = 10.0
const SHOW_LABELS = true
const DECKS = [
    ("Ground truth", joinpath(@__DIR__, "..", "models", "SPE1.DATA")),
    ("Sign constrained", joinpath(@__DIR__, "..", "models", "SPE1_sign_cons.DATA")),
    ("Sign unconstrained", joinpath(@__DIR__, "..", "models", "SPE1_sign_uncons.DATA")),
    ("Median constrained", joinpath(@__DIR__, "..", "models", "SPE1_medium-cons.DATA")),
    ("Median unconstrained", joinpath(@__DIR__, "..", "models", "SPE1_medium_uncons.DATA")),
    ("Clustering", joinpath(@__DIR__, "..", "models", "SPE1_hierarchical clustering.DATA")),
    ("No zoning", joinpath(@__DIR__, "..", "models", "spe1_no_zone.DATA")),
]

day = si_unit(:day)
bar = si_unit(:bar)
runs = NamedTuple[]
prod_wells_ref = Ref{Union{Nothing, Vector{Symbol}}}(nothing)
inj_wells_ref = Ref{Union{Nothing, Vector{Symbol}}}(nothing)

for (label, deck_path) in DECKS
    @info "Simulating deck" label = label path = deck_path
    run = run_deck(label, deck_path)
    if prod_wells_ref[] === nothing
        prod_wells_ref[] = production_wells(run.wells)
        isempty(prod_wells_ref[]) && error("No production wells found in $label")
    end
    if inj_wells_ref[] === nothing
        inj_wells_ref[] = setdiff(collect(keys(run.wells)), prod_wells_ref[])
        isempty(inj_wells_ref[]) && error("No injection wells found in $label")
    end
    missing = setdiff(prod_wells_ref[], collect(keys(run.wells)))
    !isempty(missing) && error("Deck $label is missing wells $(missing) required for aggregation")
    missing_inj = setdiff(inj_wells_ref[], collect(keys(run.wells)))
    !isempty(missing_inj) && error("Deck $label is missing injection wells $(missing_inj)")
    curves = aggregate_rates(run.wells, prod_wells_ref[])
    bhp_prod = aggregate_bhp_curve(run.wells; well_names = prod_wells_ref[]) ./ bar
    bhp_inj = aggregate_bhp_curve(run.wells; well_names = inj_wells_ref[]) ./ bar
    curves_day = Dict(q => curves[q] .* day for q in keys(curves))
    curves_day[:bhp_prod] = bhp_prod
    curves_day[:bhp_inj] = bhp_inj
    push!(runs, (;
        label = label,
        time_days = run.time ./ day,
        curves_day = curves_day,
    ))
end

# show gas, oil, water rate (grat, orat, wrat)
show_train_test_rate_comparison(runs; train_years = TRAIN_YEARS, qois = (:grat,), show_labels = SHOW_LABELS)
# show production and injection BHP (bhp_prod, bhp_inj)
show_train_test_rate_comparison(runs; train_years = TRAIN_YEARS, qois = (:bhp_inj,), show_labels = SHOW_LABELS)
