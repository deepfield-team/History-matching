function _parse_iter_loss(fields::AbstractVector{<:AbstractString})
    length(fields) ≥ 3 || return nothing
    iter_str = strip(fields[2])
    iter_str == "summary" && return nothing
    loss_str = strip(fields[3])
    iter = try
        parse(Int, iter_str)
    catch
        return nothing
    end
    loss = try
        parse(Float64, loss_str)
    catch
        return nothing
    end
    return (iter = iter, loss = loss)
end

function _parse_lbfgs_sections_by_markers(lines::Vector{String})
    labels = String[]
    iters = Vector{Vector{Int}}()
    losses = Vector{Vector{Float64}}()
    current = 0

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
            current = length(labels)
            continue
        end
        (startswith(s, "#") || startswith(s, "label")) && continue
        current > 0 || continue
        parsed = _parse_iter_loss(split(line, ","))
        parsed === nothing && continue
        push!(iters[current], parsed.iter)
        push!(losses[current], parsed.loss)
    end

    return (labels = labels, iters = iters, losses = losses)
end

function _parse_lbfgs_sections_by_label(lines::Vector{String})
    labels = String[]
    iters = Vector{Vector{Int}}()
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
        parsed = _parse_iter_loss(fields)
        parsed === nothing && continue

        idx = get!(label_to_idx, raw_label) do
            push!(labels, raw_label)
            push!(iters, Int[])
            push!(losses, Float64[])
            length(labels)
        end
        push!(iters[idx], parsed.iter)
        push!(losses[idx], parsed.loss)
    end

    return (labels = labels, iters = iters, losses = losses)
end

function load_lbfgs_sections(log_path::AbstractString)
    labels = String[]
    iters = Vector{Vector{Int}}()
    losses = Vector{Vector{Float64}}()
    isfile(log_path) || return (labels = labels, iters = iters, losses = losses)

    lines = readlines(log_path)
    by_markers = _parse_lbfgs_sections_by_markers(lines)
    if any(!isempty, by_markers.iters)
        return by_markers
    end
    return _parse_lbfgs_sections_by_label(lines)
end
