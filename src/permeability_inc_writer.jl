import Printf: Format, format

const _DEFAULT_VALUES_PER_LINE = 6
const _DEFAULT_FLOAT_FORMAT = "%12.6f"

"""
    write_permeability_inc(path; permx, permy = permx;
                           header_lines = String[],
                           values_per_line = 6,
                           float_format = "%12.6f")

Write an Eclipse-style include file that contains PERMX/PERMY decks.
PERMZ is intentionally omitted because vertical permeability is fixed elsewhere.
Input arrays must have the same size and are written with `values_per_line`
entries per row using the provided `float_format`. The include file begins
with any `header_lines` that are supplied.
"""
function write_permeability_inc(
        path::AbstractString;
        permx::AbstractArray,
        permy::AbstractArray = permx,
        header_lines::AbstractVector{<:AbstractString} = String[],
        values_per_line::Integer = _DEFAULT_VALUES_PER_LINE,
        float_format::AbstractString = _DEFAULT_FLOAT_FORMAT,
    )

    size(permy) == size(permx) ||
        throw(ArgumentError("size(permy) mismatches size(permx)"))

    open(path, "w") do io
        for line in header_lines
            println(io, line)
        end
        _write_perm_block(io, "PERMX", permx, values_per_line, float_format)
        _write_perm_block(io, "PERMY", permy, values_per_line, float_format)
    end

    return path
end

function _write_perm_block(
        io::IO,
        keyword::AbstractString,
        values::AbstractArray,
        values_per_line::Integer,
        float_format::AbstractString,
    )
    println(io, keyword)
    flat_vals = vec(values)
    fmt = Format(float_format)
    for (idx, val) in enumerate(flat_vals)
        format(io, fmt, float(val))
        if idx % values_per_line == 0
            println(io)
        else
            print(io, " ")
        end
    end
    if !isempty(flat_vals) && length(flat_vals) % values_per_line != 0
        println(io)
    end
    println(io, "/")
    println(io)
    return nothing
end
