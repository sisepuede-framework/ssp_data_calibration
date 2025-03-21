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
from utilities.utils import HelperFunctions, SSPModelForCalibration, ErrorFunctions
from utilities.diff_reports import DiffReportUtils
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
PSO_OUTPUT_PATH = build_path([OUTPUT_PATH, "pso"])
REAL_DATA_FILE_PATH = build_path([DATA_PATH, "real_data.csv"])
MISC_FILES_PATH = build_path([SRC_FILE_PATH, 'misc'])
VAR_MAPPING_FILES_PATH = build_path([MISC_FILES_PATH, 'var_mapping'])
SECTORAL_REPORT_PATH = build_path([MISC_FILES_PATH, 'sectoral_reports'])
SECTORAL_REPORT_MAPPING_PATH = build_path([MISC_FILES_PATH, 'sectoral_report_mapping'])
OPT_CONFIG_FILES_PATH = build_path([SRC_FILE_PATH, 'config'])
OPT_OUTPUT_PATH = build_path([SRC_FILE_PATH,"..", "output"])

# Make sure the output directory exists
os.makedirs(OPT_OUTPUT_PATH, exist_ok=True)
# Get important params from the YAML file

try:
    yaml_file = 'croatia_opt_config.yaml'
except IndexError:
    raise ValueError("YAML configuration file must be provided as a command-line argument.")

param_dict = helper_functions.get_parameters_from_yaml(build_path([OPT_CONFIG_FILES_PATH, yaml_file]))

target_region = param_dict['target_region']
iso_alpha_3 = param_dict['iso_alpha_3']
stressed_variables_report_version = param_dict['stressed_variables_report_version']
normalization_flag = param_dict['normalization_flag']
detailed_diff_report_flag = param_dict['detailed_diff_report_flag']
energy_model_flag = param_dict['energy_model_flag']
subsector_to_calibrate = param_dict['subsector_to_calibrate']
error_type = param_dict['error_type']
weight_type = param_dict['weight_type']
unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
swarm_size = param_dict['swarmsize']
maxiter = param_dict['maxiter']
input_rows = param_dict['input_rows']
ssp_edgar_cw_file_name = param_dict['ssp_edgar_cw']

logging.info(f"Starting optimization for {target_region} (ISO code: {iso_alpha_3})")
logging.info(f"Input rows: {input_rows}")
logging.info(f"Stressed variables report version: {stressed_variables_report_version}")
logging.info(f"Normalization flag: {normalization_flag}")
logging.info(f"Energy model flag: {energy_model_flag}")
logging.info(f"Subsector to calibrate: {subsector_to_calibrate}")
logging.info(f"Error type: {error_type}")
logging.info(f"Weight type: {weight_type}")
logging.info(f"Unique ID: {unique_id}")
logging.info(f"Swarm size: {swarm_size}")
logging.info(f"Max iterations: {maxiter}")
logging.info(f"Detailed diff report flag: {detailed_diff_report_flag}")
logging.info(f"SSP-EDGAR crosswalk file: {ssp_edgar_cw_file_name}")

# Make sure the output directory exists
os.makedirs(OPT_OUTPUT_PATH, exist_ok=True)

# Make sure pso output directories exist
os.makedirs(PSO_OUTPUT_PATH, exist_ok=True)

# Create the output directory for the PSO results using the unique ID
RUN_OUTPUT_DIR = os.path.join(PSO_OUTPUT_PATH, unique_id)
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)

# Save the config file to the output directory
config_file_name = f"{unique_id}_config.yaml"
config_file_output_path = os.path.join(RUN_OUTPUT_DIR, config_file_name)
helper_functions.copy_param_yaml(build_path([OPT_CONFIG_FILES_PATH, yaml_file]), config_file_output_path)

# Load input dataset
examples = SISEPUEDEExamples()
cr = examples("input_data_frame")
df_input = pd.read_csv(REAL_DATA_FILE_PATH)

# Add missing columns and reformat the input datas
df_input = df_input.rename(columns={'period': 'time_period'})
df_input = helper_functions.add_missing_cols(cr, df_input.copy())
df_input = df_input.drop(columns='iso_code3')

# Subset df_input to the input rows amount
df_input = df_input.iloc[:input_rows]

# Load frac_vars mapping excel
frac_vars_mapping = pd.read_excel(build_path([VAR_MAPPING_FILES_PATH, 'frac_vars_mapping.xlsx']), sheet_name='frac_vars')

# Load the stressed variables mapping file
stressed_vars_mapping = pd.read_excel(build_path([VAR_MAPPING_FILES_PATH, stressed_variables_report_version]))

# Subset the stressed variables mapping file to is_stressed = 1
stressed_vars_mapping = stressed_vars_mapping[stressed_vars_mapping['is_stressed'] == 1]

# Check for nulls in the is_stressed column
if stressed_vars_mapping['is_stressed'].isnull().sum() > 0:
    raise ValueError("There are null values in the is_stressed column of the stressed variables mapping file.")

# Reset the index of the stressed variables mapping file
stressed_vars_mapping = stressed_vars_mapping.reset_index(drop=True)

# Set group_id as integer
stressed_vars_mapping['group_id'] = stressed_vars_mapping['group_id'].astype(int)

# Check group id array
stressed_vars_mapping.group_id.unique()

# Get the list of vars to clip
vars_to_clip = stressed_vars_mapping[stressed_vars_mapping['is_capped'] == 1]['variable_name'].tolist()

# Get the frac_vars that are going to be stressed
frac_vars_to_stress = [var for var in stressed_vars_mapping['variable_name'].values if var.startswith('frac_')]

# Subset frac_vars_mapping to only include the frac_vars that are going to be stressed
frac_vars_mapping = frac_vars_mapping[frac_vars_mapping['frac_var_name'].isin(frac_vars_to_stress)].reset_index(drop=True)

# Check special_case distribution
frac_vars_mapping['special_case'].value_counts()

# Get group ids of the vars that are stressed
group_ids = stressed_vars_mapping[stressed_vars_mapping["is_stressed"] == 1]["group_id"].unique()
n_groups = len(group_ids)

# Get the lower and upper bounds for each group
l_bounds = stressed_vars_mapping.groupby("group_id")["l_bound"].first().values
u_bounds = stressed_vars_mapping.groupby("group_id")["u_bound"].first().values

# Create a dictionary with the group ids as keys and the corresponding variable names as values
group_vars_dict = {}
for group_id in group_ids:
    group_vars_dict[group_id] = stressed_vars_mapping[stressed_vars_mapping["group_id"] == group_id]["variable_name"].values

# Initialize the ErrorFunctions class
ef = ErrorFunctions()

#  Initialize the DiffReportUtils class
edgar_ssp_cw_path = build_path([SECTORAL_REPORT_MAPPING_PATH, ssp_edgar_cw_file_name])
dru = DiffReportUtils(iso_alpha_3, edgar_ssp_cw_path, SECTORAL_REPORT_PATH, energy_model_flag)

# Generate EDGAR df
edgar_emission_db_path = build_path([SECTORAL_REPORT_MAPPING_PATH, 'CSC-GHG_emissions-April2024_to_calibrate.csv'])
edgar_df = dru.edgar_emission_db_etl(edgar_emission_db_path)

# Initialize global variable to store the previous calculated error
previous_error = float('inf')

# Initialize global variable to store the worst_valid_error
worst_valid_error = float(12)

# Initialize the SSP model
ssp_model = SSPModelForCalibration(energy_model_flag=energy_model_flag)

# Simulation model
def simulation_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function that simulates outputs based on the scaled inputs.
    """
    sim_output_df = ssp_model.run_ssp_simulation(df)
    
    # Handle empty DataFrame
    if sim_output_df is None or sim_output_df.empty:
        logging.warning("Simulation Output DataFrame is empty. Returning an empty DataFrame.")
        return pd.DataFrame()

    return sim_output_df

# Define the objective function
def objective_function(x):
    
    # Global variables
    global worst_valid_error
    global previous_error
    global edgar_df
    
    
    # x: scaling factors for each group_id
    modified_df = df_input.copy()
    
    # TODO: Vectorize this loop
    # Scale the variables per group
    for group_id in group_vars_dict:
        for var in group_vars_dict[group_id]:
            modified_df[var] = modified_df[var] * x[group_id]
    
    
    if normalization_flag:
        # Handle frac var group normalization
        processed_input_df = helper_functions.simple_frac_normalization(modified_df, frac_vars_mapping)

        # Clip the variables
        processed_input_df = helper_functions.clip_values(processed_input_df, vars_to_clip)
    
    else:
        processed_input_df = modified_df.copy()

    
    # Run the model
    sim_output_df = simulation_model(processed_input_df)
    
    # Assing a penalty if the simulation output is empty
    if sim_output_df.empty:
        error_val = worst_valid_error * 1.1  # Slighly higher than the worst valid error for invalid outputs
        logging.warning("Simulation returned an empty DataFrame. Setting Error to a penalty value.")
        error_msg = "WARNING: Simulation returned an empty DataFrame."
        helper_functions.log_error_msgs(error_msg=error_msg, RUN_OUTPUT_PATH=RUN_OUTPUT_DIR, target_region=target_region, unique_id=unique_id)
    
    else:
        # Generate diff reports to calculate Error
        report_dict = dru.run_report_generator(edgar_emission_df=edgar_df, ssp_out_df=sim_output_df, subsector_to_calibrate=subsector_to_calibrate)

        # Get reports and flags from dict
        sectoral_emission_report = report_dict['sectoral_emission_report']
        subsector_emission_report = report_dict['subsector_emission_report']
        model_failed_flag = report_dict['model_failed_flag']

        
        # Checks if the model failed in any subsector
        if model_failed_flag:
            error_val = worst_valid_error * 1.1  # Slighly higher than the worst valid error for invalid outputs
            logging.warning("Model failed in a subsector. Setting Error to a penalty.")
            error_msg = "WARNING: Model failed in a subsector."
            helper_functions.log_error_msgs(error_msg=error_msg, RUN_OUTPUT_PATH=RUN_OUTPUT_DIR, target_region=target_region, unique_id=unique_id)
        
        # Calculate error
        elif detailed_diff_report_flag:
            error_val = ef.calculate_error(error_type, sectoral_emission_report, weight_type)
        else:
            error_val = ef.calculate_error(error_type, subsector_emission_report, weight_type)

    # Update worst_valid_error
    if error_val > worst_valid_error:
        worst_valid_error = error_val
        logging.info(f"New worst_valid_error: {worst_valid_error:.6f}")

    # Log the error
    logging.info("=" * 30)
    logging.info(f"Current ERROR: {error_val:.6f}")
    logging.info("=" * 30)

    # Log the scaling factors and the error
    helper_functions.log_to_csv(x, error_val, error_type, RUN_OUTPUT_DIR, target_region, unique_id)

    # Save the processed_input_df, detailed_diff_report and subsector_diff_report if the error is less than the previous error
    if error_val < previous_error:
        previous_error = error_val
        processed_input_df.to_csv(build_path([RUN_OUTPUT_DIR, f"best_input_df_{unique_id}.csv"]), index=False)
        sectoral_emission_report.to_csv(build_path([RUN_OUTPUT_DIR, f"best_detailed_diff_report_{unique_id}.csv"]), index=False)
        subsector_emission_report.to_csv(build_path([RUN_OUTPUT_DIR, f"best_subsector_diff_report_{unique_id}.csv"]), index=False)
        logging.info(f"Best Input Data and Diff Reports Updated to {RUN_OUTPUT_DIR}")

    return error_val

# Initialize the PSO optimizer
best_solution, best_value = pso(
    objective_function,
    l_bounds,
    u_bounds,
    swarmsize=swarm_size,
    maxiter=maxiter,
    )



logging.info(f"PSO optimization completed for {target_region} (run id: {unique_id})")
# Record the end time
end_time = time.time()
elapsed_time = end_time - start_time
logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
# logging.info(f"Best scaling vector: {best_solution}")
logging.info(f"Best error: {best_value}")

# Save scaling vector
scaling_vector_df = pd.DataFrame(best_solution, columns=['scaling_factor'])
scaling_vector_df.to_csv(build_path([RUN_OUTPUT_DIR, f"scaling_vector_{unique_id}.csv"]), index=False)

