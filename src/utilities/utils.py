import time
import os
import yaml
import numpy as np
import pandas as pd
import sisepuede as si

class HelperFunctions:
    """
    A collection of static helper functions for various utility tasks.
    Methods:
        print_elapsed_time(start_time):
        check_land_use_factor(ssp_object, target_region):
        add_missing_cols(df1, df2):
            Adds missing columns from df1 to df2.
        ensure_directory_exists(path):
        get_parameters_from_yaml(file_path):
        simple_frac_normalization(df, frac_vars_mapping_df):
            Normalizes fractional variables in a DataFrame based on specified mappings.
        clip_values(df, list_of_vars_to_clip, min_value=0, max_value=1):
        log_to_csv(scaling_vector, error_val, error_type, OPT_OUTPUT_PATH, target_region, unique_id):
        copy_param_yaml(file_path, new_file_path):
    """
    
    @staticmethod
    def print_elapsed_time(start_time):
        """
        Prints the elapsed time since the given start time.

        Args:
            start_time (float): The start time in seconds since the epoch (as returned by time.time()).

        Returns:
            None
        """

        # Record the end time
        end_time = time.time()

        # Calculate and print the execution time
        execution_time = end_time - start_time
        print(f"------------------------ EXECUTION TIME: {execution_time} seconds ------------------------")

    @staticmethod
    def check_land_use_factor(ssp_object, target_region):
        """
        Checks the land use reallocation factor for a given region in the SSP object.

        This function generates a scenario database from the SSP object and extracts the land use yield reallocation factor
        for the specified target region. If the sum of the 'lndu_reallocation_factor' is greater than 0, it raises a ValueError.

        Parameters:
        ssp_object (object): An instance of the SSP object that contains the scenario data and model attributes.
        target_region (str): The region for which the land use reallocation factor is to be checked.

        Raises:
        ValueError: If the sum of 'lndu_reallocation_factor' is greater than 0.
        """
        try:
            dict_scendata = ssp_object.generate_scenario_database_from_primary_key(0)
            df_inputs_check = dict_scendata.get(target_region) # Change the name of the country if running a different one
            lndu_realloc_fact_df = ssp_object.model_attributes.extract_model_variable(df_inputs_check, "Land Use Yield Reallocation Factor")
        except:
            print("Error in lndu factor...")

        if lndu_realloc_fact_df['lndu_reallocation_factor'].sum() > 0:
            raise ValueError(" --------------- The sum of 'lndu_reallocation_factor' is greater than 0 -----------------")
    
    @staticmethod
    def add_missing_cols(df1, df2):
        """
        Add missing columns from df1 to df2.
        This function identifies columns that are present in df1 but missing in df2,
        and adds those columns to df2 with their corresponding values from df1.
        Parameters:
        df1 (pandas.DataFrame): The source DataFrame containing the columns to be added.
        df2 (pandas.DataFrame): The target DataFrame to which the missing columns will be added.
        Returns:
        pandas.DataFrame: The updated df2 DataFrame with the missing columns added.
        """
        # Identify columns in df1 but not in df2
        columns_to_add = [col for col in df1.columns if col not in df2.columns]

        # Add missing columns to df2 with their values from df1
        for col in columns_to_add:
            df2[col] = df1[col]
        
        return df2
    
    @staticmethod
    def ensure_directory_exists(path):
        """
        Ensures that a directory exists at the specified path.

        If the directory does not exist, it will be created. If the directory
        already exists, a message will be printed indicating so.

        Args:
            path (str): The path to the directory to check or create.

        Returns:
            None
        """
        """Creates a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    @staticmethod
    def get_parameters_from_yaml(file_path):
        """
        Reads a YAML file and returns its contents as a dictionary.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            dict: The contents of the YAML file.
        """
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        return config
    
    @staticmethod
    def copy_param_yaml(file_path, new_file_path):
        """
        Copies the contents of a YAML file to a new file.

        Args:
            file_path (str): The path to the source YAML file.
            new_file_path (str): The path to the new YAML file.

        Returns:
            None
        """
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        with open(new_file_path, 'w') as file:
            yaml.dump(config, file)
        
        return None
    
    
    @staticmethod
    def simple_frac_normalization(df, frac_vars_mapping_df):
        """
        Normalize fractional variables in a DataFrame based on specified mappings.
        This function normalizes groups of fractional variables so that their sum equals 1
        and ensures that single fractional variables are clipped between 0 and 1.
        Parameters:
        df (pd.DataFrame): The input DataFrame containing the fractional variables to be normalized.
        frac_vars_mapping_df (pd.DataFrame): A DataFrame containing the mapping of fractional variable names
                                             and their normalization rules. It should have at least two columns:
                                             'frac_var_name_prefix' and 'special_case'.
                                             'frac_var_name_prefix' indicates the prefix of the fractional variable names.
                                             'special_case' indicates whether the variable is a single fractional variable (1)
                                             or part of a group to be normalized (0).
        Returns:
        pd.DataFrame: A new DataFrame with the normalized fractional variables.
        """

        # Copy the DataFrame to avoid modifying the original
        df_norm = df.copy()

        # Get the prefix of the frac_vars groups to normalize
        frac_vars_groups_prefix = frac_vars_mapping_df[frac_vars_mapping_df.special_case == 0]['frac_var_name_prefix'].unique()
        frac_vars_singles_prefix = frac_vars_mapping_df[frac_vars_mapping_df.special_case == 1]['frac_var_name_prefix'].unique()

        # Iterate over the prefix to normalize the frac_vars groups
        for prefix in frac_vars_groups_prefix:
            # Get the columns with the prefix
            cols = [col for col in df_norm.columns if prefix in col]

            # Normalize the columns
            df_norm[cols] = df_norm[cols].div(df_norm[cols].sum(axis=1), axis=0)

        
        # TODO: This might need to change
        # Iterate over the prefix to ensure the frac_vars singles don't exceed 1 or go below 0
        for prefix in frac_vars_singles_prefix:
            # Get the column with the prefix
            col = [col for col in df_norm.columns if prefix in col]

            # Ensure the column values are between 0 and 1
            df_norm[col] = df_norm[col].clip(0, 1)    
        
        return df_norm
    
    @staticmethod
    def clip_values(df, list_of_vars_to_clip, min_value=0, max_value=1):
        """
        Clips the values in the DataFrame to the specified range.

        Args:
            df (pd.DataFrame): The input DataFrame.
            list_of_values_to_cap (list): The list of values to cap.
            min_value (int, optional): The minimum value. Defaults to 0.
            max_value (int, optional): The maximum value. Defaults to 1.

        Returns:
            pd.DataFrame: The DataFrame with capped values.
        """
        # Copy the DataFrame to avoid modifying the original
        df_clipped = df.copy()

        # Clip the values in the DataFrame
        df_clipped[list_of_vars_to_clip] = df_clipped[list_of_vars_to_clip].clip(min_value, max_value)

        return df_clipped
    
    @staticmethod
    def log_to_csv(scaling_vector: np.ndarray, error_val: float, error_type: str, OPT_OUTPUT_PATH: str, target_region: str, unique_id: str):
        """
        Logs the scaling vector and error information to a CSV file.
        Parameters:
        scaling_vector (np.ndarray): Array of scaling factors.
        error_val (float): The error value to log.
        error_type (str): The type of error (e.g., 'MSE', 'MAE').
        OPT_OUTPUT_PATH (str): The output directory path where the CSV file will be saved.
        target_region (str): The target region identifier.
        unique_id (str): A unique identifier for the log entry.
        Returns:
        None
        """
        
        log_data = {'Error': [error_val], 'Error_Type': [error_type], **{f'scale_{i}': [val] for i, val in enumerate(scaling_vector)}}
        log_df = pd.DataFrame(log_data)
        
        # Append to the CSV file or create it if it doesn't exist
        log_file = os.path.join(OPT_OUTPUT_PATH, f"opt_results_{target_region}_{unique_id}.csv")
        log_df.to_csv(log_file, mode='a', header=not pd.io.common.file_exists(log_file), index=False)

        return None
    
    @staticmethod
    def log_error_msgs(error_msg: str, RUN_OUTPUT_PATH: str, target_region: str, unique_id: str):
        """
        Logs the error messages to a text file.
        Parameters:
        error_msg (str): The error message to log.
        OPT_OUTPUT_PATH (str): The output directory path where the text file will be saved.
        target_region (str): The target region identifier.
        unique_id (str): A unique identifier for the log entry.
        Returns:
        None
        """
        
        log_file = os.path.join(RUN_OUTPUT_PATH, f"error_msg_{target_region}_{unique_id}.txt")
        with open(log_file, 'a') as file:
            file.write(f"{error_msg}\n")
        
        return None
       

class SSPModelForCalibration:
    """
    A class to run the SSP model for calibration.
    """

    def __init__(self, energy_model_flag=False):
        """
        Initializes the SSPModelForCalibration class.
        Sets up the SISEPUEDE object and logger.
        """
        # Set logger to avoid duplicate logs
        self.log_job = None

        # Set the energy model flag
        self.energy_model_flag = energy_model_flag


        # Set up SISEPUEDE Object  
        self.ssp = si.SISEPUEDE(
            "calibrated",
            initialize_as_dummy=not(self.energy_model_flag),  # No connection to Julia is initialized if set to True
            regions=["costa_rica"],  # Dummy region to avoid errors
            db_type="csv"
        )
        
        
        self.log_job = self.ssp.logger

    def run_ssp_simulation(self, stressed_df):
        """
        Runs the SSP simulation using the provided stressed DataFrame.

        Parameters:
        stressed_df (pd.DataFrame): The input DataFrame containing stressed data.

        Returns:
        pd.DataFrame: The output DataFrame from the SSP simulation.
        """

        # Retrieve and return the output DataFrame
        try:
            # Run the SSP model projection
            df_out = self.ssp.models.project(stressed_df, include_electricity_in_energy=self.energy_model_flag)

            if df_out is None or df_out.empty:
                raise ValueError("The output DataFrame is None or empty. Returning an empty DataFrame.")
        except Exception as e:
            print(f"Warning: {e}")
            df_out = pd.DataFrame()

        return df_out
    

class ErrorFunctions:
    """
    A class containing various error calculation functions for comparing simulation values with reference values.
    Methods:
        wmse(dataframe: pd.DataFrame) -> float:
        rmse(dataframe: pd.DataFrame) -> float:
        mae(dataframe: pd.DataFrame) -> float:
        wmape(dataframe: pd.DataFrame) -> float:
        calculate_error(error_type: str, dataframe: pd.DataFrame) -> float:
    """
    
    def mse(self, dataframe):
        """
        Computes the Mean Squared Error (MSE).

        Args:
            dataframe (pd.DataFrame): DataFrame containing squared differences.

        Returns:
            float: MSE value.
        """

        # Compute MSE
        mse_value = dataframe['squared_diff'].mean()
        return mse_value
    
    
    def wmse(self, dataframe, weight_type='norm_weight'):
        """
        Computes the weighted Mean Squared Error (MSE) based on the `diff` column as weights.

        Args:
            dataframe (pd.DataFrame): DataFrame containing weights, and squared differences.

        Returns:
            float: Weighted MSE value.
        """
        # Compute WMSE
        wmse = np.sum(dataframe[weight_type] * dataframe['squared_diff']) / np.sum(dataframe[weight_type])
        return wmse
    

    def rmse(self, dataframe):
        """
        Computes the Root Mean Squared Error (RMSE).

        Args:
            dataframe (pd.DataFrame): DataFrame containing squared differences.

        Returns:
            float: RMSE value.
        """

        # Compute RMSE
        rmse_value = self.mse(dataframe) ** 0.5
        return rmse_value
    
    def mape(self, dataframe):
        """
        Computes the Mean Absolute Percentage Error (MAPE).

        Args:
            dataframe (pd.DataFrame): DataFrame containing relative errors.

        Returns:
            float: MAPE value.
        """

        # Compute MAPE
        mape = np.mean(dataframe['rel_error'].abs()) * 100
        return mape
    
    
    def wmape(self, dataframe):
        """
        Computes the weighted Mean Absolute Percentage Error (WMAPE).

        Args:
            dataframe (pd.DataFrame): DataFrame containing weights and relative errors.

        Returns:
            float: WMAPE value.
        """

        # Computer the weighted mean absolute percentage error
        wmape = np.mean(dataframe['norm_weight'] * dataframe['rel_error'].abs()) * 100
        return wmape    
    
    def calculate_error(self, error_type, dataframe, weight_type='norm_weight'):
        """
        Calculates the error based on the specified error type.

        Args:
            dataframe (pd.DataFrame): DataFrame containing weights and squared differences.

        Returns:
            float: Error value.
        """
        if error_type == 'wmse':
            return self.wmse(dataframe)
        elif error_type == 'rmse':
            return self.rmse(dataframe, weight_type=weight_type)
        elif error_type == 'mse':
            return self.mse(dataframe)
        elif error_type == 'mape':
            return self.mape(dataframe)
        elif error_type == 'wmape':
            return self.wmape(dataframe)
        else:
            raise ValueError(f"Error type '{error_type}' is not supported.")
    
    
