# Multipliers Zonation for Reservoir History Matching
This repository is the companion codebase for our paper on gradient-based zoning for reservoir history matching. It contains the full workflow we used in the study: optimization code, zonation strategies, deck variants, generated permeability include files, logs, and figure outputs used for analysis and reporting.

## Visual Overview
### How the model is divided into zones
![Zone division over refinement](pics/division.gif)

### Rates and BHP comparison across zoning strategies
![Image #1: Rates and BHP comparison for different zoning methods and no-zoning baseline](pics/compare_plot.png)

## Installation
### Supported Software Versions
- Julia: `1.12.0` (the checked-in `Manifest.toml` was generated with this version).
- OS: Linux/macOS/Windows with a working Julia installation.
- Graphics: for interactive 3D plotting (`GLMakie`), an OpenGL-capable environment is required.
- Headless environments: plotting calls may need to be disabled or adapted.

### Project Dependencies
The project is fully pinned by `Project.toml` + `Manifest.toml`. Core dependencies include:
- Reservoir simulation: `Jutul = 0.4.8`, `JutulDarcy = 0.3.0`, `GeoEnergyIO = 1.1.28`
- Optimization and refinement: `LBFGSB = 0.4.1`, `Optim = 1.13.2`, `Clustering = 0.15.8`
- Data and logs: `CSV = 0.10.15`, `DataFrames = 1.8.1`
- Visualization: `GLMakie = 0.12.0`, `CairoMakie = 0.14.0`, `Plots = 1.41.1`

### Install Steps
1. Clone the repository and enter it:
   ```bash
   git clone <your-repo-url>
   cd Gradient-based-zoning-for-reservoir-history-matching
   ```
2. Instantiate dependencies:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
   ```

## Getting Started
### 1. Run zonation-aware history matching (main workflow)
```bash
julia --project=. experiments/code_spe1_zone_LBFGS.jl
```
What this run does:
- Loads SPE1 reference data.
- Runs iterative refinement + L-BFGS multiplier optimization.
- Writes permeability include file(s) into `models/inc/`.
- Writes logs and rate curves into `experiments/logs/`.
- Displays plots (loss, gradients, permeability, rates).

### 2. Run no-zonation baseline
```bash
julia --project=. experiments/code_spe1_no_zone.jl
```
This optimizes per-cell multipliers directly and writes `models/inc/spe1_no_zone_perm.inc`.

### 3. Compare train/test behavior across deck variants
```bash
julia --project=. experiments/code_spe1_train_test_rates.jl
```
This script simulates prepared deck variants and plots train/test curves for selected QoIs.

### 4. Change refinement strategy
Edit `experiments/code_spe1_zone_LBFGS.jl`:
- Set `ZONING_TECHNIQUE_KEY` to one of: `sign_uncons`, `sign_cons`, `medium-cons`, `medium_uncons`, `hierarchical clustering`.

## Data Files: Where to Get Them
### What data is required
- Base SPE1 deck: `models/SPE1.DATA`
- Variant deck (sign constrained): `models/SPE1_sign_cons.DATA`
- Variant deck (sign unconstrained): `models/SPE1_sign_uncons.DATA`
- Variant deck (median constrained): `models/SPE1_medium-cons.DATA`
- Variant deck (median unconstrained): `models/SPE1_medium_uncons.DATA`
- Variant deck (hierarchical clustering): `models/SPE1_hierarchical clustering.DATA`
- Variant deck (no zoning): `models/spe1_no_zone.DATA`
- Permeability include files referenced by variant decks: `models/inc/*.inc`

### How data is obtained in this repository
- No manual download is needed for the checked-in decks under `models/`.
- The optimization scripts (`code_spe1_zone_LBFGS.jl`, `code_spe1_no_zone.jl`) read SPE1 through `GeoEnergyIO.test_input_file_path("SPE1")`.
- If needed, switch those scripts to the local deck path: `joinpath(@__DIR__, "..", "models", "SPE1.DATA")`.

### Regenerating include files
Running the optimization scripts regenerates include files automatically:
- Zonation workflow writes files like `models/inc/spe1_zone_LBFGS_perm_<technique>.inc`.
- No-zonation workflow writes `models/inc/spe1_no_zone_perm.inc`.

## Repository Structure
```text
.
├── src/
├── experiments/
├── models/
├── pics/
├── Project.toml
├── Manifest.toml
└── README.md
```

### `src/` (library code)
- `src/MultipliersZonation.jl`: module entry point, exports, and file includes.
- `src/loss_functions.jl`: mismatch definitions, observation builders, and auto scaling utilities.
- `src/zonation.jl`: zonation refinement strategies (sign-, median-, and clustering-based).
- `src/optimization.jl`: permeability gradient to multiplier gradient mapping.
- `src/workflows.jl`: refinement loop, optimization orchestration, history logging, field reconstruction helpers.
- `src/rate_curves.jl`: well aggregation utilities and loading of saved rate curves.
- `src/lbfgs_logs.jl`: parser for LBFGS log files and refinement sections.
- `src/decks.jl`: helper to run a deck and return standardized outputs.
- `src/permeability_inc_writer.jl`: writer for Eclipse-style `PERMX/PERMY` include files.
- `src/plotting.jl`: visualization helpers for 3D fields, losses, and rate comparisons.

### `experiments/` (entry-point scripts and outputs)
- `experiments/code_spe1_zone_LBFGS.jl`: main zonation-aware optimization pipeline.
- `experiments/code_spe1_no_zone.jl`: no-zonation baseline optimization.
- `experiments/code_spe1_train_test_rates.jl`: train/test comparison across prepared decks.

### `models/` (simulation decks and generated permeability files)
- `models/SPE1.DATA`: base ground-truth SPE1 deck.
- `models/SPE1_*.DATA` and `models/spe1_no_zone.DATA`: deck variants that include tuned permeability files.
- `models/inc/*.inc`: generated `PERMX/PERMY` include files used by the variant decks.
