module MultipliersZonationPlotting

using GLMakie
using Jutul: plot_cell_data, si_unit
using Printf: @sprintf

export show_perm_3d
export show_epoch_gradients
export show_multipliers_line
export show_multipliers_3d
export show_loss_history
export show_rate_comparison
export show_rate_comparison_with_zonation
export show_train_test_rate_comparison
export save_rate_curves!
export symrange
export show_lbfgs_sections

const DEFAULT_CASE_COLORS = (
    :steelblue,
    :darkorange,
    :seagreen,
    :firebrick,
    :mediumpurple,
    :goldenrod,
    :teal,
    :deeppink,
)

symrange(vals) = begin
    isempty(vals) && return (-1e-12, 1e-12)
    m = maximum(abs, vals)
    if isnan(m) || isinf(m) || m == 0.0
        (-1e-12, 1e-12)
    else
        (-m, m)
    end
end

function show_perm_3d(
        rmesh,
        perm_vec;
        units::Symbol = :mD,
        colormap = :viridis,
        shading = GLMakie.Makie.automatic,
        title = nothing,
        colorbar_label = nothing,
        colorrange = nothing,
    )
    fld = perm_vec[:]
    range = isnothing(colorrange) ? extrema(fld) : colorrange
    fig, ax, plt = plot_cell_data(
        rmesh, fld;
        colormap   = colormap,
        shading    = shading,
        colorrange = range,
    )
    unit_label = units === :mD ? "mD" : "m²"
    ax.title[] = isnothing(title) ? "Permeability field ($unit_label)" : String(title)
    cb_label = something(colorbar_label, @sprintf("Permeability [%s]", unit_label))
    Colorbar(fig[1, 2], plt; label = cb_label)
    display(fig)
    return fig
end

function show_epoch_gradients(
        rmesh,
        grads::AbstractVector,
        label::AbstractString;
        colormap = :balance,
        autoscale_sym::Bool = true,
    )
    fig, ax, plt = plot_cell_data(
        rmesh, grads;
        colormap = colormap,
        shading  = GLMakie.Makie.automatic,
    )
    autoscale_sym && (plt.colorrange[] = symrange(grads))
    ax.title[] = @sprintf("∂J/∂k — %s", label)
    Colorbar(fig[1, 2], plt; label = "∂J/∂k")
    display(fig)
    return fig
end

function show_multipliers_line(
        vals::AbstractVector;
        xs = 1:length(vals),
        xlabel::AbstractString = "Zone index",
        ylabel::AbstractString = "Multiplier",
        title = nothing,
    )
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel = xlabel,
        ylabel = ylabel,
        title  = something(title, "Multipliers"),
    )
    lines!(ax, xs, vals; linewidth = 2)
    scatter!(ax, xs, vals; markersize = 10)
    display(fig)
    return fig
end

function show_multipliers_3d(
        rmesh,
        mult_field::AbstractVector;
        colormap = :viridis,
        shading = GLMakie.Makie.automatic,
        title = nothing,
        colorbar_label = nothing,
        colorrange = nothing,
    )
    range = isnothing(colorrange) ? extrema(mult_field) : colorrange
    fig, ax, plt = plot_cell_data(
        rmesh, mult_field;
        colormap   = colormap,
        shading    = shading,
        colorrange = range,
    )
    ax.title[] = isnothing(title) ? "Multipliers per cell" : String(title)
    cb_label = something(colorbar_label, "Multiplier")
    Colorbar(fig[1, 2], plt; label = String(cb_label))
    display(fig)
    return fig
end

function show_loss_history(
        losses::AbstractVector,
        labels::AbstractVector{String};
        technique_name::AbstractString = "",
    )
    length(losses) == length(labels) ||
        throw(ArgumentError("losses and labels must have the same length"))
    n = length(losses)
    n > 0 || throw(ArgumentError("no loss data recorded"))
    xs = 1:n
    ttl = isempty(technique_name) ? "Loss per refinement" : @sprintf("%s — loss per refinement", technique_name)
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel = "Refinement step",
        ylabel = "Loss",
        title  = ttl,
        xticks = (xs, labels),
        xticklabelrotation = π / 4,
    )
    lines!(ax, xs, losses; linewidth = 2)
    scatter!(ax, xs, losses; markersize = 10)
    display(fig)
    return fig
end

_rate_title(qoi::Symbol) = qoi === :orat ? "Oil rate" :
                           qoi === :wrat ? "Water rate" :
                           qoi === :grat ? "Gas rate" : string(qoi)

function show_rate_comparison(
        time_truth,
        time_matched,
        curves::Dict;
        label::AbstractString = "",
    )
    day = si_unit(:day)
    t_truth = time_truth ./ day
    t_matched = time_matched ./ day
    qois = (:grat, :orat, :wrat)
    fig = Figure(size = (900, 280 * length(qois)))
    matched_label = isempty(label) ? "Matched" : label
    for (idx, qoi) in enumerate(qois)
        ttl = isempty(label) ? _rate_title(qoi) : @sprintf("%s — %s", label, _rate_title(qoi))
        ax = Axis(fig[idx, 1];
            xlabel = "Time [days]",
            ylabel = "Flow rate [m³/day]",
            title  = ttl,
        )
        truth_curve = curves[qoi].truth
        matched_curve = curves[qoi].matched
        lines!(ax, t_truth, truth_curve .* day; label = "Truth")
        lines!(ax, t_matched, matched_curve .* day; linestyle = :dash, label = matched_label)
        axislegend(ax, position = :rb)
    end
    display(fig)
    return fig
end

function show_rate_comparison_with_zonation(
        time_truth,
        curves_no_zone::Dict,
        zonation_data::NamedTuple;
        label_no_zone::AbstractString = "No zonation",
        label_zonation::AbstractString = "Zonation",
    )
    day = si_unit(:day)
    t_days = time_truth ./ day
    zn_time = getfield(zonation_data, :time_days)
    if length(zn_time) != length(t_days) || maximum(abs.(zn_time .- t_days)) > 1e-6
        @warn "Zonation rate data time vector does not match current simulation; skipping combined plot."
        return nothing
    end
    qois = (:grat, :orat, :wrat)
    fig = Figure(size = (900, 280 * length(qois)))
    for (idx, qoi) in enumerate(qois)
        ax = Axis(fig[idx, 1];
            xlabel = "Time [days]",
            ylabel = "Flow rate [m³/day]",
            title  = @sprintf("SPE1 — %s", _rate_title(qoi)),
        )
        truth_curve = curves_no_zone[qoi].truth .* day
        no_zone_curve = curves_no_zone[qoi].matched .* day
        zonation_curve = zonation_data.curves[qoi].matched
        lines!(ax, t_days, truth_curve; label = "Truth")
        lines!(ax, t_days, no_zone_curve; linestyle = :dash, label = label_no_zone)
        lines!(ax, t_days, zonation_curve; linestyle = :dot, label = label_zonation)
        axislegend(ax, position = :rb)
    end
    display(fig)
    return fig
end

function show_train_test_rate_comparison(cases;
        train_years::Real = 10.0,
        qois = (:grat, :orat, :wrat, :bhp_prod, :bhp_inj),
        case_colors = DEFAULT_CASE_COLORS,
        line_width::Real = 3,
        show_labels::Bool = false,
    )
    qoi_titles = Dict(
        :grat => "Gas rate",
        :orat => "Oil rate",
        :wrat => "Water rate",
        :bhp_prod => "BHP (PROD wells)",
        :bhp_inj => "BHP (INJ wells)",
    )
    qoi_ylabels = Dict(
        :grat => "Rate [m^3/day]",
        :orat => "Rate [m^3/day]",
        :wrat => "Rate [m^3/day]",
        :bhp_prod => "BHP [bar]",
        :bhp_inj => "BHP [bar]",
    )
    test_start = train_years * 365.0
    figs = Figure[]
    for qoi in qois
        fig = Figure(size = (1000, 320))
        ttl = get(qoi_titles, qoi, string(qoi))
        ylabel = get(qoi_ylabels, qoi, "Value")
        ax = Axis(fig[1, 1];
            xlabel = "Time [days]",
            ylabel = ylabel,
            title  = ttl,
        )
        all_vals = Float64[]
        xmax = 0.0
        for (case_idx, case) in enumerate(cases)
            vals = get(case.curves_day, qoi, nothing)
            vals === nothing && error("Missing $qoi values for case $(case.label)")
            t = case.time_days
            xmax = max(xmax, maximum(t))
            append!(all_vals, vals)
            color = case_colors[mod1(case_idx, length(case_colors))]
            line_kwargs = show_labels ? (; label = case.label) : (;)
            lines!(
                ax,
                t,
                vals;
                color = color,
                linewidth = line_width,
                line_kwargs...,
            )
        end
        ymin, ymax = isempty(all_vals) ? (0.0, 1.0) : extrema(all_vals)
        if ymax == ymin
            ymin -= 1.0
            ymax += 1.0
        end
        margin = 0.05 * (ymax - ymin)
        plot_min = ymin - margin
        plot_max = ymax + margin
        ylims!(ax, plot_min, plot_max)
        if test_start < xmax
            band!(
                ax,
                [test_start, xmax],
                fill(plot_min, 2),
                fill(plot_max, 2);
                color = (0.8, 0.8, 0.8, 0.25),
            )
            vlines!(ax, test_start; color = (:gray, 0.6), linestyle = :dot)
            text!(
                ax,
                "Training";
                position = (test_start / 2, plot_max - 0.07 * (plot_max - plot_min)),
                align = (:center, :top),
                fontsize = 14,
            )
            text!(
                ax,
                "Testing";
                position = ((test_start + xmax) / 2, plot_max - 0.07 * (plot_max - plot_min)),
                align = (:center, :top),
                fontsize = 14,
            )
        end
        show_labels && axislegend(ax; position = :rb)
        display(fig)
        push!(figs, fig)
    end
    return figs
end

function show_lbfgs_sections(
        sections::NamedTuple{(:labels, :iters, :losses)},
        ; ncols::Integer = 2,
          first_refinement::Integer = 1,
          last_refinement::Union{Integer, Nothing} = nothing,
    )
    isempty(sections.labels) && throw(ArgumentError("no LBFGS sections to plot"))

    total = length(sections.labels)
    first_idx = clamp(first_refinement, 1, total)
    last_idx = isnothing(last_refinement) ? total : clamp(last_refinement, first_idx, total)
    idxs = first_idx:last_idx

    labels = sections.labels[idxs]
    iters = sections.iters[idxs]
    losses = sections.losses[idxs]

    ncols = max(1, min(ncols, length(labels)))
    nrows = cld(length(labels), ncols)
    fig = Figure(size = (600 * ncols, 250 * nrows))
    for (idx, label) in enumerate(labels)
        row = Int(cld(idx, ncols))
        col = ((idx - 1) % ncols) + 1
        ax = Axis(fig[row, col];
            xlabel = "LBFGS iteration",
            ylabel = "Loss",
            title  = label,
            yscale = GLMakie.Makie.log10,
        )
        lines!(ax, iters[idx], losses[idx]; linewidth = 2)
        scatter!(ax, iters[idx], losses[idx]; markersize = 8)
    end
    display(fig)
    return fig
end

function save_rate_curves!(
        path::AbstractString,
        time_truth,
        curves;
        qois = (:grat, :orat, :wrat),
    )
    mkpath(dirname(path))
    day = si_unit(:day)
    t_days = time_truth ./ day
    open(path, "w") do io
        cols = ["time_days"]
        for q in qois
            push!(cols, string(q, "_truth_day"))
            push!(cols, string(q, "_matched_day"))
        end
        println(io, join(cols, ","))
        for i in eachindex(t_days)
            row = [t_days[i]]
            for q in qois
                truth_curve = curves[q].truth
                matched_curve = curves[q].matched
                push!(row, truth_curve[i] * day, matched_curve[i] * day)
            end
            println(io, join(row, ","))
        end
    end
    return path
end

end
