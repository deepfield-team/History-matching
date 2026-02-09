using GeoEnergyIO

function run_deck(label::AbstractString, deck_path::AbstractString)
    isfile(deck_path) || error("Deck not found at $deck_path")
    data = parse_data_file(deck_path)
    case = setup_case_from_data_file(data)
    result = simulate_reservoir(case)
    wells = wells_dict(result.wells)
    return (; label = String(label), time = result.time, wells)
end
