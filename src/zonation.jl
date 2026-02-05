using Statistics
using Clustering: hclust, cutree

function dummy_refine!(mults, zonation, rmesh, K_grads, mults_grads, refinement_step)
    append!(mults, mults[end])
    zonation = map(c -> min(refinement_step, cell_ijk(rmesh, c)[3]), 1:number_of_cells(rmesh))
    return zonation
end

"""
    gradient_sign_refine!(mults, zonation, K_grads; zero_tolerance = 0.0)

Split every existing zone into up to two new zones using the sign of the local gradient
(`K_grads ≥ zero_tolerance` vs `< zero_tolerance`). Each new zone inherits its multiplier
from the mean of the per-cell multipliers that are reassigned to it so the optimization
can continue smoothly.
"""
function gradient_sign_refine!(mults, zonation, K_grads; zero_tolerance = 0.0)
    @assert length(zonation) == length(K_grads)

    new_zonation = similar(zonation, Int)
    zone_sign_map = Dict{Tuple{Int, Int8}, Int}()
    next_label = 0

    @inbounds for i in eachindex(zonation)
        zone = zonation[i]
        grad = K_grads[i]
        sign_key = (!isfinite(grad) || grad ≥ zero_tolerance) ? Int8(1) : Int8(-1)
        key = (zone, sign_key)
        lbl = get!(zone_sign_map, key) do
            next_label += 1
            next_label
        end
        new_zonation[i] = lbl
    end

    _reassign_multipliers!(mults, zonation, new_zonation)
    return new_zonation
end

@inline function _sign_key(grad, zero_tolerance)
    return (!isfinite(grad) || grad ≥ zero_tolerance) ? Int8(1) : Int8(-1)
end

@inline function _asfloat(val)
    try
        return Float64(val)
    catch
        if hasfield(typeof(val), :value)
            return _asfloat(getfield(val, :value))
        end
        return Float64(val)
    end
end

function _reassign_multipliers!(
        mults,
        previous_zonation::Vector{Int},
        new_zonation::Vector{Int};
        min_multiplier::Union{Nothing, Real} = nothing,
        max_multiplier::Union{Nothing, Real} = nothing,
    )
    default_mult = isempty(mults) ? 1.0 : mults[end]
    cell_mults = isempty(mults) ? fill(default_mult, length(previous_zonation)) : mults[previous_zonation]
    nzones = maximum(new_zonation)
    zone_sums = zeros(Float64, nzones)
    zone_counts = zeros(Int, nzones)
    @inbounds for i in eachindex(new_zonation)
        z = new_zonation[i]
        zone_sums[z] += cell_mults[i]
        zone_counts[z] += 1
    end
    resize!(mults, nzones)
    @inbounds for z in 1:nzones
        val = zone_counts[z] == 0 ? default_mult : zone_sums[z] / zone_counts[z]
        if min_multiplier !== nothing
            val = max(val, min_multiplier)
        end
        if max_multiplier !== nothing
            val = min(val, max_multiplier)
        end
        mults[z] = val
    end
    return nothing
end

@inline function _zone_stats(grads, idxs, fallback_mean, fallback_median)
    vals = Float64[]
    sizehint!(vals, length(idxs))
    for idx in idxs
        v = grads[idx]
        isfinite(v) && push!(vals, v)
    end
    return isempty(vals) ? (spread = 0.0, mean = fallback_mean, median = fallback_median) :
           (spread = maximum(vals) - minimum(vals), mean = mean(vals), median = median(vals))
end

function _robust_center_scale(vals::AbstractVector{<:Real})
    med = median(vals)
    absdev = similar(vals, Float64)
    @inbounds for i in eachindex(vals)
        absdev[i] = abs(vals[i] - med)
    end
    mad = median(absdev)
    scale = 1.4826 * mad
    if !(isfinite(scale) && scale > 0)
        s = std(vals)
        scale = (isfinite(s) && s > 0) ? s : 1.0
    end
    clamp_sigma = 1.0
    scaled = similar(absdev)
    @inbounds for i in eachindex(vals)
        z = (vals[i] - med) / scale
        scaled[i] = clamp(z, -clamp_sigma, clamp_sigma)
    end
    return scaled
end

"""
    gradient_median_refine_all!(mults, zonation, K_grads)

Split every existing zone into up to two new zones using the median of the local
gradient values in each zone (greater than the median vs. the rest). This is the
median-based counterpart to `gradient_sign_refine!`, but applies the split across
all zones in a single refinement step instead of choosing a single zone to split.
Zone ids are relabeled by descending mean gradient to match the ordering strategy
used in `gradient_median_descending_refine!`.
"""
function gradient_median_refine_all!(mults, zonation, K_grads)
    @assert length(zonation) == length(K_grads)

    grads = Float64[_asfloat(g) for g in K_grads]
    finite_vals = filter(isfinite, grads)
    global_mean = isempty(finite_vals) ? 0.0 : mean(finite_vals)
    global_median = isempty(finite_vals) ? global_mean : median(finite_vals)

    zone_cells = Dict{Int, Vector{Int}}()
    @inbounds for i in eachindex(zonation)
        z = zonation[i]
        push!(get!(zone_cells, z, Int[]), i)
    end

    zone_medians = Dict{Int, Float64}()
    for (z, cells) in zone_cells
        zone_medians[z] = _zone_stats(grads, cells, global_mean, global_median).median
    end

    new_zonation = similar(zonation, Int)
    zone_side_map = Dict{Tuple{Int, Int8}, Int}()
    next_label = 0
    @inbounds for i in eachindex(zonation)
        z = zonation[i]
        g = grads[i]
        zmed = zone_medians[z]
        side = (isfinite(g) && g > zmed) ? Int8(1) : Int8(-1)
        key = (z, side)
        lbl = get!(zone_side_map, key) do
            next_label += 1
            next_label
        end
        new_zonation[i] = lbl
    end

    zone_cells_new = Dict{Int, Vector{Int}}()
    @inbounds for i in eachindex(new_zonation)
        z = new_zonation[i]
        push!(get!(zone_cells_new, z, Int[]), i)
    end
    zone_means = Dict{Int, Float64}()
    for (z, cells) in zone_cells_new
        zone_means[z] = _zone_stats(grads, cells, global_mean, global_median).mean
    end
    ordering = sort(
        collect(keys(zone_cells_new));
        lt = (a, b) -> zone_means[a] == zone_means[b] ? a < b : zone_means[a] > zone_means[b],
    )
    relabel = Dict(z => i for (i, z) in enumerate(ordering))
    @inbounds for i in eachindex(new_zonation)
        new_zonation[i] = relabel[new_zonation[i]]
    end

    _reassign_multipliers!(mults, zonation, new_zonation)
    return new_zonation
end

"""
    gradient_median_descending_refine!(...)

Median-guided refinement that keeps zone indices sorted by descending mean gradient.
"""
function gradient_median_descending_refine!(mults, zonation, K_grads)
    @assert length(zonation) == length(K_grads)

    grads = Float64[_asfloat(g) for g in K_grads]
    finite_vals   = filter(isfinite, grads)
    global_mean   = isempty(finite_vals) ? 0.0 : mean(finite_vals)
    global_median = isempty(finite_vals) ? global_mean : median(finite_vals)

    assignments = copy(zonation)
    zone_cells = Dict{Int, Vector{Int}}()
    @inbounds for i in eachindex(zonation)
        z = zonation[i]
        push!(get!(zone_cells, z, Int[]), i)
    end
    # ---------------------------------------------------------------------------

    stats = Dict{Int, NamedTuple{(:spread, :mean, :median), NTuple{3, Float64}}}()
    stat(z) = get!(stats, z) do
        _zone_stats(grads, zone_cells[z], global_mean, global_median)
    end

    target = first(keys(zone_cells))
    best = stat(target).spread
    for z in keys(zone_cells)
        s = stat(z).spread
        if s > best || (s == best && z < target)
            target, best = z, s
        end
    end

    idxs = zone_cells[target]
    if !isempty(idxs)
        zmed = stat(target).median
        next_label = maximum(keys(zone_cells))
        high, low = next_label + 1, next_label + 2
        zone_cells[high] = Int[]
        zone_cells[low]  = Int[]
        @inbounds for idx in idxs
            g = grads[idx]
            lbl = (isfinite(g) && g > zmed) ? high : low
            assignments[idx] = lbl
            push!(zone_cells[lbl], idx)
        end
        delete!(zone_cells, target)
    end

    for z in collect(keys(zone_cells))
        isempty(zone_cells[z]) && delete!(zone_cells, z)
    end
    zone_means = Dict{Int, Float64}()
    for (z, cells) in zone_cells
        zone_means[z] = _zone_stats(grads, cells, global_mean, global_median).mean
    end
    ordering = sort(collect(keys(zone_cells)), by = z -> zone_means[z], rev = true)
    relabel = Dict(z => i for (i, z) in enumerate(ordering))

    new_zonation = similar(assignments)
    @inbounds for i in eachindex(assignments)
        new_zonation[i] = relabel[assignments[i]]
    end

    _reassign_multipliers!(mults, zonation, new_zonation)
    return new_zonation
end

"""
    gradient_sign_targeted_refine!(mults, zonation, K_grads; zero_tolerance = 0.0)

Split only the zone with the highest sign heterogeneity, defined as the zone whose
fraction of positive-sign gradients is closest to 0.5. All other zones remain intact.
If the selected zone is homogeneous (all signs equal), the zonation is returned
unchanged. The sign convention matches `gradient_sign_refine!`, treating non-finite
gradients as positive.
"""
function gradient_sign_targeted_refine!(mults, zonation, K_grads; zero_tolerance = 0.0)
    @assert length(zonation) == length(K_grads)
    isempty(zonation) && return zonation

    nzones = maximum(zonation)
    pos_counts = zeros(Int, nzones)
    total_counts = zeros(Int, nzones)

    @inbounds for i in eachindex(zonation)
        z = zonation[i]
        total_counts[z] += 1
        if _sign_key(_asfloat(K_grads[i]), zero_tolerance) > 0
            pos_counts[z] += 1
        end
    end

    target = 1
    best = Inf
    for z in 1:nzones
        total = total_counts[z]
        total == 0 && continue
        p = pos_counts[z] / total
        heterogeneity = abs(p - 0.5)
        if heterogeneity < best || (heterogeneity == best && z < target)
            target = z
            best = heterogeneity
        end
    end

    if pos_counts[target] == 0 || pos_counts[target] == total_counts[target]
        return zonation
    end

    new_zonation = similar(zonation, Int)
    next_label = 0
    pos_label = 0
    neg_label = 0
    zone_label = Dict{Int, Int}()
    for z in 1:nzones
        total_counts[z] == 0 && continue
        if z == target
            pos_label = next_label + 1
            neg_label = next_label + 2
            next_label += 2
        else
            next_label += 1
            zone_label[z] = next_label
        end
    end

    @inbounds for i in eachindex(zonation)
        z = zonation[i]
        if z == target
            sign_key = _sign_key(_asfloat(K_grads[i]), zero_tolerance)
            new_zonation[i] = sign_key > 0 ? pos_label : neg_label
        else
            new_zonation[i] = zone_label[z]
        end
    end

    _reassign_multipliers!(mults, zonation, new_zonation)
    return new_zonation
end

@inline function gradient_sign_targeted_refine!(mults, zonation, K_grads, zero_tolerance::Real)
    return gradient_sign_targeted_refine!(mults, zonation, K_grads; zero_tolerance = zero_tolerance)
end

"""
    incremental_gradient_quantile_refine!(mults, zonation, rmesh, K_grads, mults_grads, refinement_step;
                                          min_multiplier = 5e-2, max_multiplier = 5.0, linkage = :ward)

Refinement that adds exactly one new zone each step using hierarchical clustering
of the signed gradient values. Gradients are robustly transformed before clustering
using a MAD-based z-score with clamping to reduce the influence of outliers. The
number of zones is derived from the current zonation (`n_prev + 1`), and the
dendrogram is cut to that count. Zone labels are ordered by ascending mean gradient.
Each new zone multiplier is initialized to the average of the previous per-cell
multipliers for its member cells and clamped to provided bounds so the next L-BFGS
run starts from a stable point.
"""
function incremental_gradient_quantile_refine!(
        mults,
        zonation,
        rmesh,
        K_grads,
        mults_grads,
        refinement_step,
        ;
        min_multiplier::Real = 5e-2,
        max_multiplier::Real = 5.0,
        linkage::Symbol = :ward,
    )
    @assert length(zonation) == length(K_grads)

    ncells = length(zonation)
    prev_nzones = max(1, length(unique(zonation)))
    nzones_target = min(prev_nzones + 1, ncells)

    raw_vals = similar(zonation, Float64)
    finite_sum = 0.0
    finite_count = 0
    @inbounds for i in eachindex(K_grads)
        g = _asfloat(K_grads[i])
        if isfinite(g)
            raw_vals[i] = g
            finite_sum += g
            finite_count += 1
        else
            raw_vals[i] = NaN
        end
    end
    fallback = finite_count == 0 ? 0.0 : finite_sum / finite_count
    @inbounds for i in eachindex(raw_vals)
        if !isfinite(raw_vals[i])
            raw_vals[i] = fallback
        end
    end
    scaled_vals = _robust_center_scale(raw_vals)

    if nzones_target == 1
        new_zonation = fill(1, ncells)
    else
        dmat = Matrix{Float64}(undef, ncells, ncells)
        @inbounds for i in 1:ncells
            dmat[i, i] = 0.0
            vi = scaled_vals[i]
            for j in i+1:ncells
                dist = abs(vi - scaled_vals[j])
                dmat[i, j] = dist
                dmat[j, i] = dist
            end
        end

        tree = hclust(dmat; linkage = linkage)
        clusters = cutree(tree; k = nzones_target)

        nzones = maximum(clusters)
        zone_sums = zeros(Float64, nzones)
        zone_counts = zeros(Int, nzones)
        @inbounds for i in eachindex(clusters)
            z = clusters[i]
            zone_sums[z] += raw_vals[i]
            zone_counts[z] += 1
        end
        zone_means = Vector{Float64}(undef, nzones)
        @inbounds for z in 1:nzones
            zone_means[z] = zone_counts[z] == 0 ? 0.0 : zone_sums[z] / zone_counts[z]
        end
        ordering = sortperm(1:nzones, by = z -> zone_means[z])
        relabel = zeros(Int, nzones)
        @inbounds for (new, old) in enumerate(ordering)
            relabel[old] = new
        end

        new_zonation = similar(zonation, Int)
        @inbounds for i in eachindex(clusters)
            new_zonation[i] = relabel[clusters[i]]
        end
    end

    _reassign_multipliers!(
        mults,
        zonation,
        new_zonation;
        min_multiplier = min_multiplier,
        max_multiplier = max_multiplier,
    )
    return new_zonation
end

"""
    zone_means(values, zonation) -> Vector{Float64}

Compute the per-zone mean of `values`, skipping non-finite entries. The helper
expects `zonation` to use 1-based zone ids and is typically used to summarize
permeability or saturation fields for any model geometry.
"""
function zone_means(values::AbstractVector, zonation::AbstractVector{<:Integer})
    length(values) == length(zonation) || throw(ArgumentError("values and zonation must have the same length"))
    nzones = maximum(zonation)
    sums = zeros(Float64, nzones)
    counts = zeros(Int, nzones)
    @inbounds for i in eachindex(values)
        zone = zonation[i]
        zone ≥ 1 || throw(ArgumentError("zone ids must be positive, got $zone"))
        val = _asfloat(values[i])
        if isfinite(val)
            sums[zone] += val
            counts[zone] += 1
        end
    end
    means = similar(sums)
    @inbounds for z in 1:nzones
        means[z] = counts[z] == 0 ? NaN : sums[z] / counts[z]
    end
    return means
end

zone_means(values::AbstractArray, zonation::AbstractVector{<:Integer}) = zone_means(vec(values), zonation)
