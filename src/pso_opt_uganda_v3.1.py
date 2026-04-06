# File: pso_opt_uganda_v3.py
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from utilities.utils import HelperFunctions, SSPModelForCalibration, ErrorFunctions
from utilities.diff_reports_v2 import DiffReportUtils
import logging
from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples
from ssp_transformations_handler.GeneralUtils import GeneralUtils
import json
import multiprocessing as mp

# NEW: imports for per-worker file structure + model
from sisepuede.manager.sisepuede_file_structure import SISEPUEDEFileStructure
import sisepuede.manager.sisepuede_models as sm

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Globals and Parameters ===
SRC_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
def build_path(PATH): return os.path.abspath(os.path.join(*PATH))
DATA_PATH = build_path([SRC_FILE_PATH, "..", "data"])
OUTPUT_PATH = build_path([SRC_FILE_PATH, "..", "output"])
MISC_FILES_PATH = build_path([SRC_FILE_PATH, 'misc'])
VAR_MAPPING_FILES_PATH = build_path([MISC_FILES_PATH, 'var_mapping'])
SECTORAL_REPORT_PATH = build_path([MISC_FILES_PATH, 'sectoral_reports'])
SECTORAL_REPORT_MAPPING_PATH = build_path([MISC_FILES_PATH, 'sectoral_report_mapping'])
OPT_CONFIG_FILES_PATH = build_path([SRC_FILE_PATH, 'config'])
OPT_OUTPUT_PATH = build_path([SRC_FILE_PATH,"..", "output"])

os.makedirs(OPT_OUTPUT_PATH, exist_ok=True)

# === Config and Inputs ===
yaml_file = 'uganda_opt_config.yaml'
helper_functions = HelperFunctions()
param_dict = helper_functions.get_parameters_from_yaml(build_path([OPT_CONFIG_FILES_PATH, yaml_file]))

target_region = param_dict['target_region']
iso_alpha_3 = param_dict['iso_alpha_3']
stressed_variables_report_version = param_dict['stressed_variables_report_version']
input_data_file_to_calibrate = param_dict["input_data_file_to_calibrate"]
detailed_diff_report_flag = param_dict['detailed_diff_report_flag']
energy_model_flag = param_dict['energy_model_flag']
emission_targets_file = param_dict['emission_targets_file']
sim_init_year = param_dict['sim_init_year']
filter_start_year = param_dict['filter_start_year']
filter_end_year = param_dict['filter_end_year']
comparison_year = param_dict['comparison_year']
subsector_to_calibrate = param_dict['subsector_to_calibrate']
error_type = param_dict['error_type']
weight_type = param_dict['weight_type']
unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
swarm_size = param_dict['swarmsize']
maxiter = param_dict['maxiter']
ssp_edgar_cw_file_name = param_dict['ssp_edgar_cw']

# NEW: read parallelism from YAML (fallback to 1)
n_processes = int(param_dict.get('n_processes', 1))

PSO_OUTPUT_PATH = build_path([OUTPUT_PATH, target_region])
os.makedirs(PSO_OUTPUT_PATH, exist_ok=True)
RUN_OUTPUT_DIR = os.path.join(PSO_OUTPUT_PATH, unique_id)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
config_file_output_path = os.path.join(RUN_OUTPUT_DIR, f"{unique_id}_config.yaml")
helper_functions.copy_param_yaml(build_path([OPT_CONFIG_FILES_PATH, yaml_file]), config_file_output_path)

# === Data Loading ===
REAL_DATA_FILE_PATH = build_path([DATA_PATH, input_data_file_to_calibrate])
examples = SISEPUEDEExamples()
cr = examples("input_data_frame")
df_input = pd.read_csv(REAL_DATA_FILE_PATH)
df_input = df_input.rename(columns={'period': 'time_period'})
gu = GeneralUtils()
df_input = gu.add_missing_cols(cr, df_input.copy())
df_input = df_input.drop(columns='iso_code3', errors='ignore')
time_period_init_year = filter_start_year - sim_init_year
time_period_end_year = filter_end_year - sim_init_year
df_input = df_input[(df_input['time_period'] >= time_period_init_year) & (df_input['time_period'] <= time_period_end_year)].reset_index(drop=True)

# === Variable Mapping ===
stressed_vars_mapping = pd.read_excel(build_path([VAR_MAPPING_FILES_PATH, stressed_variables_report_version]))
stressed_vars_mapping = stressed_vars_mapping[stressed_vars_mapping['is_stressed'] == 1].reset_index(drop=True)
stressed_vars_mapping['group_id'] = stressed_vars_mapping['group_id'].astype(int)
stressed_vars_mapping = stressed_vars_mapping.sort_values(by='group_id', ascending=True)
group_ids = stressed_vars_mapping["group_id"].unique()
n_groups = len(group_ids)
group_vars_dict = {group_id: stressed_vars_mapping[stressed_vars_mapping["group_id"] == group_id]["variable_name"].values for group_id in group_ids}
reordered_dict = {new_id: group_vars_dict[old_id] for new_id, old_id in enumerate(group_vars_dict, 0)}
l_bounds = stressed_vars_mapping.groupby("group_id")["l_bound"].first().values
u_bounds = stressed_vars_mapping.groupby("group_id")["u_bound"].first().values

ef = ErrorFunctions()
edgar_ssp_cw_path = build_path([SECTORAL_REPORT_MAPPING_PATH, ssp_edgar_cw_file_name])
dru = DiffReportUtils(iso_alpha_3, edgar_ssp_cw_path, SECTORAL_REPORT_PATH, energy_model_flag, sim_init_year=sim_init_year, comparison_year=comparison_year)
edgar_emission_db_path = build_path([SECTORAL_REPORT_MAPPING_PATH, 'emission_targets_uganda.csv'])
edgar_df = dru.get_edgar_region_df(edgar_emission_db_path)
if edgar_df.empty:
    raise ValueError(f"EDGAR DataFrame is empty. Please check that your iso_alpha_3 is in {edgar_emission_db_path}")

previous_error = float('inf')
worst_valid_error = float(12)

# === Per-process model registry (NEW) ===
_MODEL_REGISTRY = {}            # {worker_slot: model_object}
_REGISTRY_LOCK = mp.Lock()

def _get_worker_slot(n_workers: int) -> int:
    """Stable index [0..n_workers-1] for the current Pool worker."""
    name = mp.current_process().name  # e.g., "ForkPoolWorker-1" / "SpawnPoolWorker-1"
    try:
        if "-" in name and name.split("-")[-1].isdigit():
            return (int(name.split("-")[-1]) - 1) % max(1, n_workers)
    except Exception:
        pass
    # Fallback on pid modulus if name parse fails
    return os.getpid() % max(1, n_workers)

def build_calibration_model(temp_sqlite_path: str):
    """
    Build and return a model instance that will run simulations using a worker-unique temp SQLite DB.

    >>> IMPORTANT: pick ONE of the two implementations below (A or B) that matches your APIs.
    Then delete the other one to avoid confusion.
    """

    # ---------------------------
    # (A) If SSPModelForCalibration can accept an injected SISEPUEDEModels object:
    # ---------------------------
    # fs = SISEPUEDEFileStructure()
    # models_cur = sm.SISEPUEDEModels(
    #     # NOTE: replace `ssp_model_attributes_source` with however you get model_attributes
    #     ssp_model_attributes_source,  # e.g., examples("model_attributes") or similar
    #     fp_julia=fs.dir_jl,
    #     fp_nemomod_reference_files=fs.dir_ref_nemo,
    #     fp_nemomod_temp_sqlite_db=temp_sqlite_path,
    # )
    # return SSPModelForCalibration(
    #     energy_model_flag=energy_model_flag,
    #     external_models=models_cur  # <-- this parameter must exist in your class
    # )

    # ---------------------------
    # (B) If SSPModelForCalibration takes a direct override for the temp sqlite path:
    # ---------------------------
    return SSPModelForCalibration(
        energy_model_flag=energy_model_flag,
        fp_nemomod_temp_sqlite_db=temp_sqlite_path  # <-- this parameter must exist in your class
    )

def _get_or_create_model(worker_slot: int):
    """Create (once) and return a worker-local model bound to a unique temp SQLite DB."""
    with _REGISTRY_LOCK:
        if worker_slot in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[worker_slot]

        fs = SISEPUEDEFileStructure()
        # Base template from your file structure; give it a worker-specific suffix
        base_tmp = fs.fp_sqlite_tmp_nemomod_intermediate
        tmp_sqlite = base_tmp.replace(".sqlite", f"_prcs_{worker_slot}.sqlite")

        model = build_calibration_model(tmp_sqlite)
        _MODEL_REGISTRY[worker_slot] = model
        logging.info(f"[Worker {worker_slot}] Initialized model with temp DB: {tmp_sqlite}")
        return model

# === Main Simulation and Objective ===
def simulation_model(df: pd.DataFrame) -> pd.DataFrame:
    # pull the process-local model bound to a unique DB
    slot = _get_worker_slot(n_processes)
    model = _get_or_create_model(slot)

    sim_output_df = model.run_ssp_simulation(df)
    if sim_output_df is None or sim_output_df.empty:
        logging.warning("Simulation Output DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()
    return sim_output_df

def objective_function(x):
    global worst_valid_error
    global previous_error
    global edgar_df
    global reordered_dict

    modified_df = df_input.copy()
    for group_id in reordered_dict:
        for var in reordered_dict[group_id]:
            if var in modified_df.columns:
                modified_df[var] = modified_df[var] * x[group_id]
            else:
                logging.warning(f"Variable '{var}' not found in input DataFrame. Skipping scaling for this variable.")
    processed_input_df = modified_df.copy()
    sim_output_df = simulation_model(processed_input_df)
    if sim_output_df.empty:
        error_val = worst_valid_error * 1.1
        helper_functions.log_error_msgs(
            error_msg="WARNING: Simulation returned an empty DataFrame.",
            RUN_OUTPUT_PATH=RUN_OUTPUT_DIR, target_region=target_region, unique_id=unique_id)
        sectoral_emission_report = pd.DataFrame()
        subsector_emission_report = pd.DataFrame()
    else:
        report_dict = dru.run_report_generator(edgar_emission_df=edgar_df, ssp_out_df=sim_output_df)
        sectoral_emission_report = report_dict['sectoral_emission_report']
        subsector_emission_report = report_dict['subsector_emission_report']
        model_failed_flag = report_dict['model_failed_flag']
        if model_failed_flag:
            error_val = worst_valid_error * 1.1
            helper_functions.log_error_msgs(
                error_msg="WARNING: Model failed in a subsector.",
                RUN_OUTPUT_PATH=RUN_OUTPUT_DIR, target_region=target_region, unique_id=unique_id)
        elif detailed_diff_report_flag:
            error_val = ef.calculate_error(error_type, sectoral_emission_report, weight_type, subsector_to_calibrate)
        else:
            error_val = ef.calculate_error(error_type, subsector_emission_report, weight_type, subsector_to_calibrate)

    if error_val > worst_valid_error:
        worst_valid_error = error_val
        logging.info(f"New worst_valid_error: {worst_valid_error:.6f}")

    logging.info("=" * 30)
    logging.info(f"Current ERROR: {error_val:.6f}")
    logging.info("=" * 30)

    helper_functions.log_to_csv(x, error_val, error_type, RUN_OUTPUT_DIR, target_region, unique_id)

    if sim_output_df is not None and not sim_output_df.empty and error_val < previous_error:
        previous_error = error_val
        processed_input_df.to_csv(build_path([RUN_OUTPUT_DIR, f"best_input_df_{unique_id}.csv"]), index=False)
        sectoral_emission_report.to_csv(build_path([RUN_OUTPUT_DIR, f"best_detailed_diff_report_{unique_id}.csv"]), index=False)
        subsector_emission_report.to_csv(build_path([RUN_OUTPUT_DIR, f"best_subsector_diff_report_{unique_id}.csv"]), index=False)
        logging.info(f"Best Input Data and Diff Reports Updated to {RUN_OUTPUT_DIR}")

    return error_val

def vectorized_objective(x: np.ndarray) -> np.ndarray:
    return np.array([objective_function(x_i) for x_i in x])

if __name__ == "__main__":
    # Strongly recommend spawn on macOS to avoid forking Julia
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set in this interpreter; that's fine
        pass

    start_time = time.time()
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    dimensions = len(l_bounds)

    optimizer = GlobalBestPSO(
        n_particles=swarm_size,
        dimensions=dimensions,
        options=options,
        bounds=(l_bounds, u_bounds)
    )

    best_cost, best_pos = optimizer.optimize(
        vectorized_objective,
        iters=maxiter,
        n_processes=n_processes  # NEW: match our worker-slot logic
    )
    logging.info(f"PSO optimization completed for {target_region} (run id: {unique_id})")
    elapsed_time = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info(f"Best error: {best_cost}")

    # Save scaling vector
    scaling_vector_df = pd.DataFrame({'group_id': np.arange(len(best_pos)), 'scaling_factor': best_pos})
    scaling_vector_df.to_csv(build_path([RUN_OUTPUT_DIR, f"scaling_vector_{unique_id}.csv"]), index=False)

    # Save the reordered dictionary to a JSON file
    serializable_dict = {k: v.tolist() for k, v in reordered_dict.items()}
    with open(build_path([RUN_OUTPUT_DIR, f"reordered_dict_{unique_id}.json"]), 'w') as json_file:
        json.dump(serializable_dict, json_file, indent=2)

    logging.info(f"Best solution length: {len(best_pos)}")
    logging.info(f"Reordered dictionary keys length: {len(serializable_dict.keys())}")

    # Plot the cost history
    from pyswarms.utils.plotters import plot_cost_history
    plot_cost_history(optimizer.cost_history)
    plt.savefig(build_path([RUN_OUTPUT_DIR, f"cost_history_{unique_id}.png"]))
