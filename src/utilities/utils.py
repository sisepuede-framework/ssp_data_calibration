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
    

class SectoralDiffReport:
    
    
    def __init__(self, sectoral_report_dir_path, iso_alpha_3, init_year, ref_year=2015):
        
        # Set up variables
        self.iso_alpha_3 = iso_alpha_3
        self.ref_year = ref_year # year to calibrate on
        self.init_year = init_year # Simulation's start year
        self.sectoral_report_dir_path = sectoral_report_dir_path
        self.report_type = 'all-sectors'
        self.model_failed_flag = False

    def load_mapping_table(self, file_path):
       
        # Load mapping tables
        mapping_df = pd.read_csv(file_path)
        
        return mapping_df
    
    def load_simulation_output_data(self, simulation_df):
        """
        Filters the simulation output data for the reference year.
        This method takes a DataFrame containing simulation output data, adds a 'year' column
        based on the 'time_period' column and the initial year, and then filters the DataFrame
        to include only the rows corresponding to the reference year.
        Args:
            simulation_df (pd.DataFrame): The DataFrame containing the simulation output data.
                It must have a 'time_period' column.
        Returns:
            pd.DataFrame: A filtered DataFrame containing only the rows for the reference year.
        """

        simulation_df_filtered = simulation_df.copy()

        # Add a year column to the simulation data
        simulation_df_filtered['year'] = simulation_df_filtered['time_period'] + self.init_year

        # Check if ref_year is present in the simulation data
        if self.ref_year not in simulation_df_filtered['year'].unique():
            raise ValueError(f"The reference year {self.ref_year} is not present in the simulation data. Please increase your range of simulated years.")

        # Filter the simulation data to the reference year
        simulation_df_filtered = simulation_df_filtered[simulation_df_filtered['year'] == self.ref_year]
 
        return simulation_df_filtered
    
    def edgar_data_etl(self, file_path):
       
        # Load Edgar data
        edgar_df = pd.read_csv(file_path, encoding='latin1')

        # Filter Edgar data to the region of interest
        edgar_df = edgar_df[edgar_df['Code'] == self.iso_alpha_3].reset_index(drop=True)

        # Create Edgar_Class column by combining Subsector and Gas columns
        edgar_df['Edgar_Class'] = edgar_df['CSC Subsector'] + ':' + edgar_df['Gas']

        # Specify the id_vars (columns to keep) and value_vars (columns to unpivot)
        
        id_vars = ['Edgar_Class']
        value_vars = [str(self.ref_year)]

        # Melt the DataFrame
        edgar_df_long = edgar_df.melt(id_vars=id_vars, value_vars=value_vars, 
                        var_name='Year', value_name='Edgar_Values')

        # Convert the 'year' column to integer type
        edgar_df_long['Year'] = edgar_df_long['Year'].astype(int)
        
        return edgar_df_long
    
    def load_emission_targets(self, file_path):

        # Load emissions target df
        targets_df = pd.read_csv(file_path)

        # Filter the targets_df to relevant columns
        targets_df = targets_df[['Subsector', 'Gas', 'Edgar_Class', 'Year', self.iso_alpha_3]]

        # Rename the iso_alpha_3 column to 'Edgar_Values'
        targets_df.rename(columns={self.iso_alpha_3: 'Edgar_Values'}, inplace=True)

        return targets_df

    
    def calculate_ssp_emission_totals(self, simulation_df, mapping_df):
        """
        Calculate the total emissions for each subsector based on the provided simulation data.
        This function takes a simulation DataFrame and a mapping DataFrame, and calculates the total
        emissions for each subsector defined in the mapping DataFrame. The results are stored in a new
        column 'Simulation_Values' in the mapping DataFrame.
        Args:
            simulation_df (pd.DataFrame): DataFrame containing the simulation data with various emission variables.
            mapping_df (pd.DataFrame): DataFrame containing the mapping of subsectors to emission variables.
        Returns:
            pd.DataFrame: Updated mapping DataFrame with a new column 'Simulation_Values' containing the total emissions
                          for each subsector.
        Notes:
            - The 'Vars' column in the mapping DataFrame should contain colon-separated variable names.
            - If any variables listed in the 'Vars' column are not present in the simulation DataFrame, they will be reported.
        """
        
        # Create detailed report df from mapping df
        detailed_report_draft_df = mapping_df.copy()
        
        # Add a new column to store total emissions
        detailed_report_draft_df['Simulation_Values'] = 0  # Initialize with zeros

        # Create a set of all column names in simulation_df for quick lookup
        simulation_df_cols = set(simulation_df.columns)

        # Create a set to store all missing variable names
        missing_variables = set()

        # Iterate through each row in detailed_report_draft_df
        for index, row in detailed_report_draft_df.iterrows():
            vars_list = row['Vars'].split(':')  # Split Vars column into variable names

            # Check which variable names are missing in simulation_df
            missing_in_row = [var for var in vars_list if var not in simulation_df_cols]
            missing_variables.update(missing_in_row)  # Add missing variables to the set

            # Filter the columns in simulation_df that match the variable names
            matching_columns = [col for col in vars_list if col in simulation_df_cols]

            if matching_columns:
                # Sum the matching columns across all rows in simulation_df
                subsector_total_emissions = simulation_df[matching_columns].sum().sum()
            else:
                subsector_total_emissions = 0  # No matching columns found

            # Update the simulation_values column in detailed_report_draft_df
            detailed_report_draft_df.at[index, 'Simulation_Values'] = subsector_total_emissions

        # Print missing variable names, if any
        if missing_variables:
            # print("The following variables from Vars are not present in simulation_df:")
            # for var in missing_variables:
            #     print(var)
            self.model_failed_flag = True
        else:
            self.model_failed_flag = False
            # print("All variables from Vars are present in simulation_df.")

        # Returns the updated detailed_report_draft_df
        return detailed_report_draft_df
    
    def calculate_difference(self, diff_report_draft_df):
        """
        Calculate the difference and squared difference between simulation values and Edgar values.

        Args:
            diff_report_draft_df (pd.DataFrame): DataFrame containing 'Simulation_Values' and 'Edgar_Values' columns.

        Returns:
            pd.DataFrame: DataFrame with additional columns 'diff' and 'squared_diff' representing the relative difference
                          and squared difference between 'Simulation_Values' and 'Edgar_Values', respectively.
        """

        df = diff_report_draft_df.copy()
        df['diff'] = (df['Simulation_Values'] - df['Edgar_Values']) / df['Edgar_Values']
        df['squared_diff'] = (df['Edgar_Values'] - df['Simulation_Values']) ** 2
        return df
    
    
    def generate_detailed_diff_report(self, detailed_report_draft_df, edgar_df):
        """
        Generates a detailed difference report by comparing simulation values with Edgar values.
        Args:
            detailed_report_draft_df (pd.DataFrame): DataFrame containing the draft detailed report with simulation values.
            edgar_df (pd.DataFrame): DataFrame containing the Edgar values for comparison.
        Returns:
            pd.DataFrame: DataFrame containing the year, subsector, Edgar class, simulation values, Edgar values, and the calculated difference.
        """

        detailed_diff_report = detailed_report_draft_df.copy()

        # Group by Subsector and Edgar_Class and aggregate the Simulation_Values to match Edgar_Values format
        detailed_diff_report_agg = detailed_diff_report.groupby(['Subsector', 'Edgar_Class'])['Simulation_Values'].sum().reset_index()

        # Merge the aggregated DataFrame with the Edgar data
        detailed_diff_report_merge = pd.merge(detailed_diff_report_agg, edgar_df, how='left', on=['Subsector', 'Gas', 'Edgar_Class'])

        # Calculate the difference between Simulation_Values and Edgar_Values
        detailed_diff_report_merge['diff'] = (detailed_diff_report_merge['Simulation_Values'] - detailed_diff_report_merge['Edgar_Values']) / detailed_diff_report_merge['Edgar_Values']

        # Reset Year column to ref year to avoid NaN values
        detailed_diff_report_merge['Year'] = self.ref_year

        detailed_diff_report_complete = detailed_diff_report_merge[['Year', 'Subsector', 'Gas', 'Edgar_Class', 'Simulation_Values', 'Edgar_Values', 'diff']]
        
        return detailed_diff_report_complete
    
    def generate_subsector_diff_report(self, detailed_diff_report_complete):
        """
        Generates a subsector difference report by comparing simulation values with Edgar values.
        Args:
            detailed_diff_report_complete (pd.DataFrame): DataFrame containing detailed difference report with columns 
                                                          'Subsector', 'Simulation_Values', and 'Edgar_Values'.
        Returns:
            pd.DataFrame: A DataFrame containing the subsector difference report with columns 'Year', 'Subsector', 
                          'Simulation_Values', 'Edgar_Values', and 'diff'. The 'diff' column represents the difference 
                          between 'Simulation_Values' and 'Edgar_Values' as a fraction of 'Edgar_Values'.
        """
        
        # Group by Subsector and calculate the sum of the Simulation_Values and Edgar_Values
        subsector_diff_report = detailed_diff_report_complete.groupby('Subsector')[['Simulation_Values', 'Edgar_Values']].sum().reset_index()

        # Calculate the difference between Simulation_Values and Edgar_Values
        subsector_diff_report['diff'] = (subsector_diff_report['Simulation_Values'] - subsector_diff_report['Edgar_Values']) / subsector_diff_report['Edgar_Values']

        # Reset Year column to ref year to avoid NaN values
        subsector_diff_report['Year'] = self.ref_year

        # Reorder columns
        subsector_diff_report = subsector_diff_report[['Year', 'Subsector', 'Simulation_Values', 'Edgar_Values', 'diff']] 

        return subsector_diff_report
    
    def calculate_weights(self, detailed_diff_report_complete):
        """
        Calculate weights based on the absolute values of 'Edgar_Values' column.

        Parameters:
        detailed_diff_report_complete (pd.DataFrame): DataFrame containing the 'Edgar_Values' column.

        Returns:
        pd.DataFrame: DataFrame with an additional 'Weights' column where each weight is calculated as the 
                      absolute value of 'Edgar_Values' divided by the sum of absolute values of 'Edgar_Values'.
        """
        df = detailed_diff_report_complete.copy()
        df['Weights'] = df['Edgar_Values'].abs() / df['Edgar_Values'].abs().sum()
        return df
    
    def run_report_generator(self, simulation_df, emission_targets_path, mapping_table_path):
        """
        Generates and saves detailed and subsector difference reports based on the provided simulation data.
        Args:
            simulation_df (pd.DataFrame): DataFrame containing the simulation output data.
        Returns:
            tuple: A tuple containing:
                - detailed_diff_report_complete (pd.DataFrame): DataFrame with the complete detailed difference report.
                - subsector_diff_report (pd.DataFrame): DataFrame with the subsector difference report.
        Steps:
            1. Load the mapping table.
            2. Filter the simulation output data.
            3. Perform ETL on EDGAR data.
            4. Calculate SSP emission totals.
            5. Generate the detailed difference report.
            6. Generate the subsector difference report.
            7. Save the reports to CSV files.
        """

        mapping_df = self.load_mapping_table(file_path=mapping_table_path)
        simulation_df_filtered = self.load_simulation_output_data(simulation_df)
        edgar_df = self.load_emission_targets(file_path=emission_targets_path)
        detailed_report_draft_df = self.calculate_ssp_emission_totals(simulation_df_filtered, mapping_df)
        detailed_diff_report_complete = self.generate_detailed_diff_report(detailed_report_draft_df, edgar_df)
        subsector_diff_report = self.generate_subsector_diff_report(detailed_diff_report_complete)

        detailed_diff_report_complete.to_csv(os.path.join(self.sectoral_report_dir_path, f'detailed_diff_report_{self.report_type}.csv'), index=False)
        subsector_diff_report.to_csv(os.path.join(self.sectoral_report_dir_path, f'subsector_diff_report_{self.report_type}.csv'), index=False)

        return detailed_diff_report_complete, subsector_diff_report


class NonEnergySectoralDiffReport(SectoralDiffReport):
    """
    A class to generate a sectoral difference report for non-energy sectors.
    Attributes:
        non_energy_subsectors (list): A list of non-energy subsectors.
        report_type (str): The type of report, set to 'non-energy-sectors'.
    Methods:
        __init__(sectoral_report_dir_path, iso_alpha_3, init_year, ref_year=2015):
            Initializes the NonEnergySectoralDiffReport with the given parameters.
        calculate_ssp_emission_totals(simulation_df, mapping_df):
            Calculates the SSP emission totals for non-energy subsectors based on the provided simulation and mapping dataframes.
            Args:
                simulation_df (pd.DataFrame): DataFrame containing simulation data.
                mapping_df (pd.DataFrame): DataFrame containing mapping data.
            Returns:
                pd.DataFrame: Updated DataFrame with total emissions for non-energy subsectors.
    """

    def __init__(self, sectoral_report_dir_path, iso_alpha_3, init_year, ref_year=2015):
        """
        Initializes the utility class for handling sectoral reports.

        Args:
            sectoral_report_dir_path (str): The directory path where sectoral reports are stored.
            iso_alpha_3 (str): The ISO Alpha-3 code representing the country.
            init_year (int): The initial year for the report data.
            ref_year (int, optional): The reference year for the report data. Defaults to 2015.

        Attributes:
            non_energy_subsectors (list): A list of non-energy subsectors.
            report_type (str): The type of report, set to 'non-energy-sectors'.
        """
        super().__init__(sectoral_report_dir_path, iso_alpha_3, init_year, ref_year)
        self.non_energy_subsectors = ['agrc', 'frst', 'ippu', 'lndu', 'lsmm', 'lvst', 'soil', 'trww', 'waso']
        self.report_type = 'non-energy-sectors'

    def calculate_ssp_emission_totals(self, simulation_df, mapping_df):
        """
        Calculate the total emissions for non-energy subsectors based on the provided simulation data.
        This method processes the `mapping_df` to filter out non-energy subsectors and then calculates
        the total emissions for each subsector by summing the relevant columns in `simulation_df`.
        It also identifies and reports any variables listed in the `Vars` column of `mapping_df` that
        are not present in `simulation_df`.
        Parameters:
        - simulation_df (pd.DataFrame): DataFrame containing the simulation data with various emission variables.
        - mapping_df (pd.DataFrame): DataFrame containing the mapping information, including subsectors and variable names.
        Returns:
        - pd.DataFrame: Updated DataFrame with total emissions for each non-energy subsector in the `Simulation_Values` column.
        """
        
        # Create detailed report df from mapping df
        detailed_report_draft_df = mapping_df.copy()

        # Filter the detailed_report_draft_df to non-energy subsectors
        detailed_report_draft_df = detailed_report_draft_df[detailed_report_draft_df['Subsector'].isin(self.non_energy_subsectors)]

        # Add a new column to store total emissions
        detailed_report_draft_df['Simulation_Values'] = 0  # Initialize with zeros

        # Create a set of all column names in simulation_df for quick lookup
        simulation_df_cols = set(simulation_df.columns)

        # Create a set to store all missing variable names
        missing_variables = set()

        # Iterate through each row in detailed_report_draft_df
        for index, row in detailed_report_draft_df.iterrows():
            vars_list = row['Vars'].split(':')  # Split Vars column into variable names

            # Check which variable names are missing in simulation_df
            missing_in_row = [var for var in vars_list if var not in simulation_df_cols]
            missing_variables.update(missing_in_row)  # Add missing variables to the set

            # Filter the columns in simulation_df that match the variable names
            matching_columns = [col for col in vars_list if col in simulation_df_cols]

            if matching_columns:
                # Sum the matching columns across all rows in simulation_df
                subsector_total_emissions = simulation_df[matching_columns].sum().sum()
            else:
                subsector_total_emissions = 0  # No matching columns found

            # Update the simulation_values column in detailed_report_draft_df
            detailed_report_draft_df.at[index, 'Simulation_Values'] = subsector_total_emissions

        # Print missing variable names, if any
        if missing_variables:
            # print("The following variables from Vars are not present in simulation_df:")
            # for var in missing_variables:
            #     print(var)
            self.model_failed_flag = True
        else:
            # print("All variables from Vars are present in simulation_df.")
            self.model_failed_flag = False
            
        # Returns the updated detailed_report_draft_df
        return detailed_report_draft_df


class ErrorFunctions:
    """
    A class containing various error calculation functions for comparing simulation values with reference values.
    Methods:
        wmse(dataframe: pd.DataFrame) -> float:
        rmse(dataframe: pd.DataFrame) -> float:
        mae(dataframe: pd.DataFrame) -> float:
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
        else:
            raise ValueError(f"Error type '{error_type}' is not supported.")
