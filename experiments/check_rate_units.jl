import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Jutul, JutulDarcy, GeoEnergyIO

const SPE1_WELLS = [:PROD, :INJ]

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

function main()
    data_pth = joinpath(GeoEnergyIO.test_input_file_path("SPE1"), "SPE1.DATA")
    data = parse_data_file(data_pth)
    case_truth = setup_case_from_data_file(data)
    result = simulate_reservoir(case_truth)
    wells = wells_dict(result.wells)
    prod = production_wells(wells)
    day = si_unit(:day)

    for qoi in (:orat, :grat, :wrat)
        vals = Float64[]
        for w in prod
            append!(vals, abs.(Float64.(wells[w][qoi])))
        end
        max_raw = maximum(vals)
        max_day = max_raw * day
        println("$qoi max: raw=$(max_raw), per_day=$(max_day)")
    end
end

main()
