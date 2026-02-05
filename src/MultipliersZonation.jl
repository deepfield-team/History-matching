module MultipliersZonation

export dummy_refine!
export gradient_sign_refine!
export gradient_sign_targeted_refine!
export gradient_median_refine_all!
export gradient_median_descending_refine!
export incremental_gradient_quantile_refine!
export multipliers_gradients!
export case_gradients
export LossFunctionRegistry
export RateObservations
export build_rate_observations
export set_state_reference!
export set_pressure_reference!
export set_rate_observations!
export compute_auto_scales
export bhp_mismatch
export rates_mismatch
export reservoir_pressure_mismatch
export default_loss_registry
export zone_means
export write_permeability_inc
export LBFGS_DEFAULTS
export RefinementHistory
export log_refinement!
export history_length
export loss_from_registry
export reset_lbfgs_log_file!
export log_lbfgs_history
export optimize_multipliers!
export run_refinement_loop!
export multiplier_field_for_refinement
export perm_field
export perm_field_for_refinement
export wells_dict
export production_wells
export aggregate_rates
export aggregate_bhp_curve
export collect_rate_curves
export load_zonation_rate_curves
export load_lbfgs_sections
export run_deck
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

using Jutul, JutulDarcy

include("loss_functions.jl")
include("zonation.jl")
include("optimization.jl")
include("workflows.jl")
include("rate_curves.jl")
include("lbfgs_logs.jl")
include("decks.jl")
include("permeability_inc_writer.jl")
include("plotting.jl")
using .MultipliersZonationPlotting

end
