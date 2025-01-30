import copy
import datetime as dt
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pathlib
import sys
import time
import pickle
from typing import Union
import warnings
from datetime import datetime
from pyswarm import pso  # Install with: pip install pyswarm
warnings.filterwarnings("ignore")
from info_grupos import empirical_vars_to_avoid, frac_vars_special_cases_list
from utilities.utils import HelperFunctions, SSPModelForCalibration, SectoralDiffReport, NonEnergySectoralDiffReport, ErrorFunctions
import logging
from sisepuede.manager.sisepuede_examples import SISEPUEDEExamples


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Record the start time
start_time = time.time()

# Initialize helper functions
helper_functions = HelperFunctions()


# Paths
SRC_FILE_PATH = os.getcwd()
build_path = lambda PATH: os.path.abspath(os.path.join(*PATH))
DATA_PATH = build_path([SRC_FILE_PATH, "..", "data"])
OUTPUT_PATH = build_path([SRC_FILE_PATH, "..", "output"])
REAL_DATA_FILE_PATH = build_path([DATA_PATH, "real_data.csv"])
MISC_FILES_PATH = build_path([SRC_FILE_PATH, 'misc'])
OPT_CONFIG_FILES_PATH = build_path([SRC_FILE_PATH, 'config'])
OPT_OUTPUT_PATH = build_path([SRC_FILE_PATH,"..", "output"])

# Make sure the output directory exists
os.makedirs(OPT_OUTPUT_PATH, exist_ok=True)

# Get important params from the YAML file

try:
    yaml_file = sys.argv[1]
except IndexError:
    raise ValueError("YAML configuration file must be provided as a command-line argument.")

param_dict = helper_functions.get_parameters_from_yaml(build_path([OPT_CONFIG_FILES_PATH, yaml_file]))

target_region = param_dict['target_region']
iso_alpha_3 = param_dict['iso_alpha_3']
detailed_diff_report_flag = param_dict['detailed_diff_report_flag']
error_type = param_dict['error_type'] 
unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

logging.info(f"Starting optimization for {target_region} (ISO code: {iso_alpha_3}). Detailed diff report: {'enabled' if detailed_diff_report_flag else 'disabled'}, Error type: {error_type}.")

# Load input dataset
examples = SISEPUEDEExamples()
cr = examples("input_data_frame")

df_input = pd.read_csv(REAL_DATA_FILE_PATH)
df_input = df_input.rename(columns={'period': 'time_period'})
df_input = helper_functions.add_missing_cols(cr, df_input.copy())
df_input = df_input.drop(columns='iso_code3')

# Columns to scale
columns_all_999 = df_input.columns[(df_input == -999).any()].tolist()
pij_cols = [col for col in df_input.columns if col.startswith('pij')]
cols_to_avoid = pij_cols + frac_vars_special_cases_list + columns_all_999 + empirical_vars_to_avoid
cols_to_stress = helper_functions.get_indicators_col_names(df_input, cols_with_issue=cols_to_avoid)

# Define bounds for scaling
n_vars = len(cols_to_stress)
lb = np.ones(n_vars) * param_dict['l_bound']
ub = np.ones(n_vars) * param_dict['u_bound']

logging.info(f'Bounds for scaling: {lb[0]} to {ub[0]}')

# Simulation model
def simulation_model(df_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Function that simulates outputs based on the scaled inputs.
    """
    sim_output_df = ssp_model.run_ssp_simulation(df_scaled)
    
    # Handle empty DataFrame
    if sim_output_df is None or sim_output_df.empty:
        logging.warning("Simulation Output DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    return sim_output_df

# Objective function
def objective_function(scaling_vector: np.ndarray) -> float:
    """
    Evaluates the error between simulated and ground truth outputs.
    """
    stressed_df = df_input.copy()
    stressed_df[cols_to_stress] = df_input[cols_to_stress] * scaling_vector

    # Normalize fractional columns
    normalized_stressed_df = helper_functions.normalize_frac_vars(stressed_df, cols_to_avoid, MISC_FILES_PATH)

    # Simulate outputs
    simulated_outputs = simulation_model(normalized_stressed_df)
    
    # Handle empty simulation outputs
    if simulated_outputs.empty:
        error_val = 1e6  # Assign a high Error for garbage outputs
        logging.warning("Simulation returned an empty DataFrame. Setting Error to a high value.")
    
    else:
        # Generate diff reports to calculate Error
        detailed_diff_report, subsector_diff_report = diff_report_helpers.generate_diff_reports(simulation_df=simulated_outputs)

        # Calculate error: Subsectors with Edgar value == 0.0 are not considered
        if detailed_diff_report_flag:
            error_val = ef.calculate_error(error_type, detailed_diff_report)
        else:
            error_val = ef.calculate_error(error_type, subsector_diff_report)

    logging.info("=" * 30)
    logging.info(f"Current ERROR: {error_val:.6f}")
    logging.info("=" * 30)

    # Log the results
    log_to_csv(scaling_vector, error_val)
    return error_val

# Function to log Error, error type, and scaling vector to a CSV file
def log_to_csv(scaling_vector: np.ndarray, error_val: float):
    """
    Logs the Error, error type, and scaling vector to a CSV file.
    """
    log_data = {'Error': [error_val], 'Error_Type': [error_type], **{f'scale_{i}': [val] for i, val in enumerate(scaling_vector)}}
    log_df = pd.DataFrame(log_data)
    
    # Append to the CSV file or create it if it doesn't exist
    log_file = build_path([OPT_OUTPUT_PATH, f"opt_results_{target_region}_{unique_id}.csv"])
    try:
        log_df.to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)
        logging.info(f"Logged current Error, error type, and scaling vector to {log_file}")
    except Exception as e:
        logging.error(f"Error logging data to CSV: {e}")

logging.info("Initializing SSP model...")
ssp_model = SSPModelForCalibration()
diff_report_helpers = NonEnergySectoralDiffReport(MISC_FILES_PATH, iso_alpha_3, init_year=2015)
ef = ErrorFunctions()

logging.info("Starting optimization...")
# Run PSO to find optimal scaling vector
best_scaling_vector, best_error = pso(
    objective_function,  # Objective function
    lb,  # Lower bounds
    ub,  # Upper bounds
    swarmsize = 50,  # Number of particles in the swarm
    maxiter = 100,  # Maximum iterations
    debug=True  # Display progress
)

logging.info("Optimization complete.")
helper_functions.print_elapsed_time(start_time)

# Save the best scaling vector and its Error value
results = np.append(best_scaling_vector, best_error)  # Append the error value to the scaling vector
header = ','.join([f"scale_{i}" for i in range(len(best_scaling_vector))]) + ',error'  # Create a header
output_file = build_path([OPT_OUTPUT_PATH, "best_scaling_vector.csv"])  # Save under opt_output
np.savetxt(output_file, [results], delimiter=",", header=header, comments="")  # Save with header
logging.info(f"Best scaling vector: {best_scaling_vector}")
logging.info(f"Best error: {best_error}")
