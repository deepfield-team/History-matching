# multipliers-zonation
History matching by different kinds of the permeability multipliers zonation

In the `src` folder:
- `zonation.jl` -
- `optimization.jl` -
- `loss_functions.jl` -
- `workflows.jl` -
- `rate_curves.jl` -
- `decks.jl` -
- `lbfgs_logs.jl` -
- `permeability_inc_writer.jl` -
- `plotting.jl` - shared plotting helpers for experiments

Example of running pipeline and place for experiments `experiments` folder
- `spe1.jl` - example of permeability multipliers optimization and zonation for `spe1` benchmark

To run new experiment setup zonation method in file `zonation.jl`. Then add `export new_zonation_method` in `MultipliersZonation.jl` module definition
