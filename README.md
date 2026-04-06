# ssp_data_calibration

Calibration utilities for SISEPUEDE input data. This repository runs Particle Swarm Optimization (PSO) over groups of SISEPUEDE input variables so simulated emissions better match a reference emissions inventory, such as EDGAR-derived targets.

The main workflow is:

1. Load a country input CSV from `data/`.
2. Load a stressed-variable mapping workbook from `src/misc/var_mapping/`.
3. Build PSO decision variables from mapping `group_id` values and their lower/upper bounds.
4. Scale each variable group in the input data by the candidate PSO scaling vector.
5. Run the scaled input data through SISEPUEDE.
6. Convert SISEPUEDE outputs into sector/subsector emissions reports using crosswalk files in `src/misc/sectoral_report_mapping/`.
7. Compare simulated emissions to reference targets and compute an error metric.
8. Save the best scaled input data, diff reports, scaling vector, and run configuration in `output/`.

## Repository Contents

```text
.
├── data/
│   ├── croatia_input_data.csv
│   ├── input_ssp_uganda_250522.csv
│   ├── new_uganda_inputs.csv
│   └── ssp_inputs_uganda_ASP_Scenario.csv
├── environment.yml
├── src/
│   ├── config/
│   │   ├── croatia_opt_config.yaml
│   │   └── uganda_opt_config.yaml
│   ├── misc/
│   │   ├── dummy/
│   │   ├── sectoral_report_mapping/
│   │   ├── sectoral_reports/
│   │   └── var_mapping/
│   ├── utilities/
│   │   ├── diff_reports_v2.py
│   │   └── utils.py
│   ├── pso_opt_uganda_v3.py
│   ├── pso_opt_uganda.py
│   ├── pso_opt.py
│   └── *.ipynb
└── output/
```

Important paths:

- `environment.yml`: Conda environment definition. It installs Python 3.11, Jupyter support, plotting dependencies, SISEPUEDE from a pinned Git commit, `ssp-transformations-handler`, and `pyswarms`.
- `data/`: Country input datasets to calibrate.
- `src/config/`: YAML files that select the country, input data, target data, PSO settings, error metric, and reporting options.
- `src/misc/var_mapping/`: Excel workbooks that define which variables are stressed, how they are grouped, and the PSO bounds for each group.
- `src/misc/sectoral_report_mapping/`: Crosswalks and target emissions files used to compare SISEPUEDE output to external reference data.
- `src/misc/sectoral_reports/`: Example or intermediate sectoral reports.
- `src/misc/dummy/`: Small dummy input/output/report examples.
- `src/utilities/utils.py`: Helper functions, SISEPUEDE model wrapper, YAML loading, logging, and error functions.
- `src/utilities/diff_reports_v2.py`: Report-generation utilities that clean SISEPUEDE outputs, map them to emissions categories, merge against reference targets, compute deviations, and produce sector/subsector reports.
- `output/`: Generated optimization output. This directory is ignored by git.

## Setup

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate opt_env
```

If the environment already exists and `environment.yml` changes, update it with:

```bash
conda env update -f environment.yml --prune
```

Optional, register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name opt_env --display-name "Python (opt_env)"
```

## Running an Optimization

Run commands from the repository root unless otherwise noted.

Recommended Uganda workflow:

```bash
python src/pso_opt_uganda_v3.py
```

Legacy single-process Uganda workflow:

```bash
python src/pso_opt_uganda.py
```

Legacy Croatia workflow:

```bash
python src/pso_opt.py
```

Notes:

- `src/pso_opt_uganda_v3.py` uses `pyswarms` and reads `src/config/uganda_opt_config.yaml`.
- `src/pso_opt_uganda.py` is an older Uganda script that uses the `pyswarm` API. The current `environment.yml` installs `pyswarms`; install `pyswarm` separately before using this legacy path.
- `src/pso_opt.py` is the older Croatia-oriented script and reads `src/config/croatia_opt_config.yaml`. Check dependencies and imports before using it as the active workflow.
- The numbered Uganda variants are development iterations. The Uganda script family is the best starting point for adding a new country.

## Configuration

The optimization scripts read YAML files from `src/config/`.

`src/config/uganda_opt_config.yaml` currently includes:

```yaml
target_region: uganda
iso_alpha_3: UGA
stressed_variables_report_version: stressed_variables_report_2025_06_06.xlsx
input_data_file_to_calibrate: new_uganda_inputs.csv
emission_targets_file: emission_targets_uganda.csv
detailed_diff_report_flag: true
energy_model_flag: false
sim_init_year: 2015
filter_start_year: 2015
filter_end_year: 2017
comparison_year: 2015
subsector_to_calibrate:
error_type: mse
weight_type: norm_weight
ssp_edgar_cw: sisepuede_edgar_active_crosswalk.csv
swarmsize: 468
maxiter: 2
n_processes: 2
```

Common fields:

- `target_region`: Output subdirectory name under `output/`.
- `iso_alpha_3`: ISO 3166-1 alpha-3 country code used to select target emissions columns.
- `stressed_variables_report_version`: Excel workbook in `src/misc/var_mapping/` that defines stressed variables, groups, and bounds.
- `input_data_file_to_calibrate`: CSV file in `data/` used as the base SISEPUEDE input data.
- `emission_targets_file`: Reference emissions target file used by the Uganda workflow.
- `detailed_diff_report_flag`: If `true`, calculate the optimization error from the detailed sectoral report. If `false`, calculate from the subsector report.
- `energy_model_flag`: Passed into the SISEPUEDE wrapper. When `false`, energy/electricity categories such as `fgtv`, `entc`, and `ccsq` are excluded from some report checks.
- `sim_init_year`: Calendar year corresponding to SISEPUEDE `time_period == 0`.
- `filter_start_year` and `filter_end_year`: Calendar-year window used to subset the Uganda input data before optimization.
- `comparison_year`: Calendar year used to compare simulated emissions against reference emissions.
- `subsector_to_calibrate`: Optional subsector filter for the error calculation. Leave blank to use all available subsectors.
- `error_type`: Error metric. Supported values are `mse`, `wmse`, `rmse`, `mape`, and `wmape`.
- `weight_type`: Weight column used by weighted metrics, usually `norm_weight`, `direct_weight`, or `log_weight`.
- `ssp_edgar_cw`: Crosswalk CSV in `src/misc/sectoral_report_mapping/`.
- `swarmsize`: Number of particles in the PSO run.
- `maxiter`: Number of PSO iterations.
- `n_processes`: Number of worker processes used by experimental parallel variants that read this field. `src/pso_opt_uganda_v3.py` currently uses a hard-coded worker count.

## Variable Mapping Workbooks

The stressed-variable report workbook controls the optimization dimensions. Rows with `is_stressed == 1` are included in the PSO search.

Expected columns include:

- `variable_name`: SISEPUEDE input variable to scale.
- `is_stressed`: `1` to include the variable in calibration, otherwise excluded.
- `group_id`: Variables with the same group are scaled by the same PSO decision variable.
- `l_bound`: Lower bound for the group scaling factor.
- `u_bound`: Upper bound for the group scaling factor.
- `is_capped`: Used by older normalization/clipping code paths to keep selected variables inside `[0, 1]`.

The script sorts stressed variables by `group_id`, builds one PSO dimension per unique group, then creates a `reordered_dict` that maps PSO vector positions to the variable names scaled by that position.

## Emissions Reports

`src/utilities/diff_reports_v2.py` is responsible for translating SISEPUEDE output into calibration reports:

- It converts SISEPUEDE `time_period` values into calendar years using `sim_init_year`.
- It filters simulation output to `comparison_year`.
- It loads the SSP-to-EDGAR crosswalk from `ssp_edgar_cw`.
- It sums SISEPUEDE output variables listed in the crosswalk into `ssp_emission`.
- It merges simulated emissions against the target emissions file.
- It adds `edgar_emission_epsilon`, `rel_error`, and `squared_diff`.
- It calculates `direct_weight`, `norm_weight`, and `log_weight`.
- It returns both a detailed sectoral report and a grouped subsector report.

## Output Files

Each run creates a timestamped output directory:

```text
output/<target_region>/<YYYYMMDDHHMMSS>/
```

Typical output files:

- `<run_id>_config.yaml`: Copy of the YAML configuration used for the run.
- `opt_results_<target_region>_<run_id>.csv`: Per-evaluation log of scaling factors and error values.
- `best_input_df_<run_id>.csv`: Best scaled input dataset found so far.
- `best_detailed_diff_report_<run_id>.csv`: Best detailed sectoral comparison report.
- `best_subsector_diff_report_<run_id>.csv`: Best subsector comparison report.
- `scaling_vector_<run_id>.csv`: Final best scaling vector.
- `reordered_dict_<run_id>.json`: Mapping from scaling-vector index to affected variables.
- `cost_history_<run_id>.png`: Cost-history plot for `pyswarms` workflows.
- `error_msg_<target_region>_<run_id>.txt`: Logged warnings for failed or empty simulations, when applicable.

Because `output/` is ignored by git, run results are local artifacts unless explicitly copied elsewhere.

## Adding a New Country

Use the Uganda workflow as the template for new country calibrations.

1. Add the country input CSV to `data/`.
2. Add or update a target emissions file in `src/misc/sectoral_report_mapping/`.
3. Confirm the country has a column in the target emissions file matching its `iso_alpha_3`.
4. Confirm the crosswalk in `src/misc/sectoral_report_mapping/` maps the SISEPUEDE output variables needed for your target categories.
5. Create or update a stressed-variable workbook in `src/misc/var_mapping/`.
6. Add a YAML config in `src/config/` with the new country settings.
7. Copy the Uganda script or parameterize it to read the new YAML file.
8. Start with small `swarmsize`, low `maxiter`, and a short year window to validate the data flow.
9. Inspect `best_detailed_diff_report_*`, `best_subsector_diff_report_*`, and `opt_results_*`.
10. Increase `swarmsize`, `maxiter`, and year coverage once the run produces valid reports.

## Notebooks

The notebooks under `src/` are exploratory helpers for running PSO, generating SISEPUEDE output files, testing diff-report utilities, and evaluating optimization outputs:

- `src/pso_opt.ipynb`
- `src/pso_opt_uganda.ipynb`
- `src/create_ssp_output_file.ipynb`
- `src/test_diff_report_class.ipynb`
- `src/opt_evaluation.ipynb`

Use the Conda environment as the notebook kernel.

## Troubleshooting

- If imports fail, confirm `conda activate opt_env` was run and that the `pip` dependencies from `environment.yml` installed successfully.
- If a run cannot find a data file, check that YAML file names are relative to `data/`, `src/misc/var_mapping/`, or `src/misc/sectoral_report_mapping/` as appropriate.
- If the EDGAR or target DataFrame is empty, confirm `iso_alpha_3` exists as a column in the selected target emissions file.
- If `comparison_year` is missing from simulated output, increase the configured input year range or adjust `sim_init_year`, `filter_start_year`, and `filter_end_year`.
- If SISEPUEDE returns an empty output, the optimizer logs a penalty error and appends a message to the run error log.
- If parallel runs fail on macOS or with Julia-related resources, reduce `n_processes` or use the single-process Uganda workflow while debugging.

## License

This repository is licensed under the terms in `LICENSE`.
