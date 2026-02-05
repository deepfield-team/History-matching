function _parse_lbfgs_sections_by_markers(lines::Vector{String})
    labels = String[]
    iters  = Vector{Vector{Int}}()
    losses = Vector{Vector{Float64}}()
    current = Ref(0)
    for line in lines
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "# section")
            parts = split(line, ",", limit = 2)
            length(parts) == 2 || continue
            label = strip(parts[2])
            push!(labels, label)
            push!(iters, Int[])
            push!(losses, Float64[])
            current[] = length(labels)
            continue
        end
        (startswith(s, "#") || startswith(s, "label")) && continue
        current[] > 0 || continue
        fields = split(line, ",")
        length(fields) ≥ 3 || continue
        iter_str = strip(fields[2])
        iter_str == "summary" && continue
        loss_str = strip(fields[3])
        iter = try
            parse(Int, iter_str)
        catch
            continue
        end
        loss = try
            parse(Float64, loss_str)
        catch
            continue
        end
        push!(iters[current[]], iter)
        push!(losses[current[]], loss)
    end
    return (labels = labels, iters = iters, losses = losses)
end

function _parse_lbfgs_sections_by_label(lines::Vector{String})
    labels = String[]
    iters  = Vector{Vector{Int}}()
    losses = Vector{Vector{Float64}}()
    label_to_idx = Dict{String, Int}()
    for line in lines
        s = strip(line)
        isempty(s) && continue
        (startswith(s, "#") || startswith(s, "label")) && continue
        fields = split(line, ",")
        length(fields) ≥ 3 || continue
        raw_label = strip(fields[1])
        isempty(raw_label) && continue
        iter_str = strip(fields[2])
        iter_str == "summary" && continue
        loss_str = strip(fields[3])
        iter = try
            parse(Int, iter_str)
        catch
            continue
        end
        loss = try
            parse(Float64, loss_str)
        catch
            continue
        end
        idx = get!(label_to_idx, raw_label) do
            push!(labels, raw_label)
            push!(iters, Int[])
            push!(losses, Float64[])
            length(labels)
        end
        push!(iters[idx], iter)
        push!(losses[idx], loss)
    end
    return (labels = labels, iters = iters, losses = losses)
end

function load_lbfgs_sections(log_path::AbstractString)
    labels = String[]
    iters  = Vector{Vector{Int}}()
    losses = Vector{Vector{Float64}}()
    isfile(log_path) || return (labels = labels, iters = iters, losses = losses)

    lines = readlines(log_path)
    by_markers = _parse_lbfgs_sections_by_markers(lines)
    if any(!isempty, by_markers.iters)
        return by_markers
    end
    return _parse_lbfgs_sections_by_label(lines)
end
