import time
import os
import yaml
import numpy as np
import pandas as pd
import sisepuede as si

class HelperFunctions:
    
    def __init__(self) -> None:
        pass

    
    def print_elapsed_time(self, start_time):

        # Record the end time
        end_time = time.time()

        # Calculate and print the execution time
        execution_time = end_time - start_time
        print(f"------------------------ EXECUTION TIME: {execution_time} seconds ------------------------")

    def check_land_use_factor(self, ssp_object, target_region):
        try:
            dict_scendata = ssp_object.generate_scenario_database_from_primary_key(0)
            df_inputs_check = dict_scendata.get(target_region) # Change the name of the country if running a different one
            lndu_realloc_fact_df = ssp_object.model_attributes.extract_model_variable(df_inputs_check, "Land Use Yield Reallocation Factor")
        except:
            print("Error in lndu factor...")

        if lndu_realloc_fact_df['lndu_reallocation_factor'].sum() > 0:
            raise ValueError(" --------------- The sum of 'lndu_reallocation_factor' is greater than 0. Script terminated. -----------------")

    def add_missing_cols(self, df1, df2):
        # Identify columns in df1 but not in df2
        columns_to_add = [col for col in df1.columns if col not in df2.columns]

        # Add missing columns to df2 with their values from df1
        for col in columns_to_add:
            df2[col] = df1[col]
        
        return df2

    def ensure_directory_exists(self, path):
        """Creates a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    
    def get_parameters_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)

        return config
    
    
    @staticmethod
    def simple_frac_normalization(df, frac_vars_mapping_df):

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
    
    
    def normalize_frac_vars(self, stressed_df, cols_to_avoid, misc_files_path):

        df = stressed_df.copy()

        # Normalizing frac_ var groups using softmax
        df_frac_vars = pd.read_excel(os.path.join(misc_files_path, 'frac_vars.xlsx'), sheet_name='frac_vars_no_special_cases')
        need_norm_prefix = df_frac_vars.frac_var_name_prefix.unique()

        random_scale = 1e-2  # Scale for random noise
        epsilon = 1e-6

        for subgroup in need_norm_prefix:
            subgroup_cols = [i for i in df.columns if subgroup in i]
            
            # Skip normalization for columns in cols_to_avoid
            if any(col in cols_to_avoid for col in subgroup_cols):
                continue

            # Check if the sum of the group is zero or too small
            group_sum = df[subgroup_cols].sum(axis=1)
            is_zero_sum = group_sum < epsilon

            # Add random variability for zero-sum groups
            if is_zero_sum.any():
                noise = np.random.uniform(0, random_scale, size=(is_zero_sum.sum(), len(subgroup_cols)))
                df.loc[is_zero_sum, subgroup_cols] = noise

            # Apply softmax normalization
            df[subgroup_cols] = df[subgroup_cols].apply(
                lambda row: np.exp(row) / np.exp(row).sum(), axis=1
            )

        # Special case for ce_problematic
        ce_problematic = [
            'frac_waso_biogas_food',
            'frac_waso_biogas_sludge',
            'frac_waso_biogas_yard',
            'frac_waso_compost_food',
            'frac_waso_compost_methane_flared',
            'frac_waso_compost_sludge',
            'frac_waso_compost_yard'
        ]

        # Apply softmax normalization for ce_problematic
        df[ce_problematic] = df[ce_problematic].apply(
            lambda row: np.exp(row) / np.exp(row).sum(), axis=1
        )

        return df
    


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
            df_out = self.ssp.models.project(stressed_df.head(), include_electricity_in_energy=self.energy_model_flag)

            if df_out is None or df_out.empty:
                raise ValueError("The output DataFrame is None or empty. Returning an empty DataFrame.")
        except Exception as e:
            print(f"Warning: {e}")
            df_out = pd.DataFrame()

        return df_out
    

class SectoralDiffReport:
    
    def __init__(self, misc_dir_path, iso_alpha_3, init_year, ref_year=2015):
        """
        Initialize the utility class with the given parameters.
        Args:
            misc_dir_path (str): The directory path where miscellaneous files are stored.
            iso_alpha_3 (str): The ISO 3166-1 alpha-3 country code.
            init_year (int): The initial year for the simulation.
            ref_year (int, optional): The reference year. Defaults to 2015.
        Attributes:
            iso_alpha_3 (str): The ISO 3166-1 alpha-3 country code.
            ref_year (int): The reference year.
            mapping_table_path (str): The path to the mapping table CSV file.
            init_year (int): The initial year for the simulation.
            edga_file_path (str): The path to the EDGAR data file containing ground truth data.
            misc_dir_path (str): The directory path where miscellaneous files are stored.
            report_type (str): The type of report, default is 'all-sectors'.
        """
        
        # Set up variables
        self.iso_alpha_3 = iso_alpha_3
        self.ref_year = ref_year # Reference year
        self.mapping_table_path = os.path.join(misc_dir_path, 'mapping.csv') # Mapping table path
        self.init_year = init_year # Simulation's start year
        self.edga_file_path = os.path.join(misc_dir_path, 'CSC-GHG_emissions-April2024_to_calibrate.csv') # Edgar data file path containing ground truth data
        self.misc_dir_path = misc_dir_path
        self.report_type = 'all-sectors'

    def load_mapping_table(self):
        # Load mapping tables
        mapping_df = pd.read_csv(self.mapping_table_path)
        
        return mapping_df
    
    def load_simulation_output_data(self, simulation_df):

        simulation_df_filtered = simulation_df.copy()

        # Add a year column to the simulation data
        simulation_df_filtered['year'] = simulation_df_filtered['time_period'] + self.init_year

        # Filter the simulation data to the reference year and reference primary id
        simulation_df_filtered = simulation_df_filtered[simulation_df_filtered['year'] == self.ref_year]
 
        return simulation_df_filtered
    
    def edgar_data_etl(self):
        # Load Edgar data
        edgar_df = pd.read_csv(self.edga_file_path, encoding='latin1')

        # Filter Edgar data to the reference year and reference primary id
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

    
    def calculate_ssp_emission_totals(self, simulation_df, mapping_df):
        
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
            print("The following variables from Vars are not present in simulation_df:")
            for var in missing_variables:
                print(var)
        else:
            print("All variables from Vars are present in simulation_df.")

        # Returns the updated detailed_report_draft_df
        return detailed_report_draft_df
    
    def generate_detailed_diff_report(self, detailed_report_draft_df, edgar_df):

        detailed_diff_report = detailed_report_draft_df.copy()

        # Group by Subsector and Edgar_Class and aggregate the Simulation_Values to match Edgar_Values format
        detailed_diff_report_agg = detailed_diff_report.groupby(['Subsector', 'Edgar_Class'])['Simulation_Values'].sum().reset_index()

        # Merge the aggregated DataFrame with the Edgar data
        detailed_diff_report_merge = pd.merge(detailed_diff_report_agg, edgar_df, how='left', left_on='Edgar_Class', right_on='Edgar_Class')

        # Calculate the difference between Simulation_Values and Edgar_Values
        detailed_diff_report_merge['diff'] = (detailed_diff_report_merge['Simulation_Values'] - detailed_diff_report_merge['Edgar_Values']) / detailed_diff_report_merge['Edgar_Values']

        # Reset Year column to ref year to avoid NaN values
        detailed_diff_report_merge['Year'] = self.ref_year

        detailed_diff_report_complete = detailed_diff_report_merge[['Year', 'Subsector', 'Edgar_Class', 'Simulation_Values', 'Edgar_Values', 'diff']]
        
        return detailed_diff_report_complete
    
    def generate_subsector_diff_report(self, detailed_diff_report_complete):
        
        # Group by Subsector and calculate the sum of the Simulation_Values and Edgar_Values
        subsector_diff_report = detailed_diff_report_complete.groupby('Subsector')[['Simulation_Values', 'Edgar_Values']].sum().reset_index()

        # Calculate the difference between Simulation_Values and Edgar_Values
        subsector_diff_report['diff'] = (subsector_diff_report['Simulation_Values'] - subsector_diff_report['Edgar_Values']) / subsector_diff_report['Edgar_Values']

        # Reset Year column to ref year to avoid NaN values
        subsector_diff_report['Year'] = self.ref_year

        # Reorder columns
        subsector_diff_report = subsector_diff_report[['Year', 'Subsector', 'Simulation_Values', 'Edgar_Values', 'diff']] 

        return subsector_diff_report
    
    def generate_diff_reports(self, simulation_df):

        mapping_df = self.load_mapping_table()
        simulation_df_filtered = self.load_simulation_output_data(simulation_df)
        edgar_df = self.edgar_data_etl()
        detailed_report_draft_df = self.calculate_ssp_emission_totals(simulation_df_filtered, mapping_df)
        detailed_diff_report_complete = self.generate_detailed_diff_report(detailed_report_draft_df, edgar_df)
        subsector_diff_report = self.generate_subsector_diff_report(detailed_diff_report_complete)

        detailed_diff_report_complete.to_csv(os.path.join(self.misc_dir_path, f'detailed_diff_report_{self.report_type}.csv'), index=False)
        subsector_diff_report.to_csv(os.path.join(self.misc_dir_path, f'subsector_diff_report_{self.report_type}.csv'), index=False)

        return detailed_diff_report_complete, subsector_diff_report


class NonEnergySectoralDiffReport(SectoralDiffReport):

    def __init__(self, misc_dir_path, iso_alpha_3, init_year, ref_year=2015):
        super().__init__(misc_dir_path, iso_alpha_3, init_year, ref_year)
        self.non_energy_subsectors = ['agrc', 'frst', 'ippu', 'lndu', 'lsmm', 'lvst', 'soil', 'trww', 'waso']
        self.report_type = 'non-energy-sectors'

    def calculate_ssp_emission_totals(self, simulation_df, mapping_df):
        
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
            print("The following variables from Vars are not present in simulation_df:")
            for var in missing_variables:
                print(var)
        else:
            print("All variables from Vars are present in simulation_df.")

        # Returns the updated detailed_report_draft_df
        return detailed_report_draft_df


class ErrorFunctions:
    
    def weighted_mse(self, dataframe):
        """
        Computes the weighted Mean Squared Error (MSE) based on the `diff` column as weights.

        Args:
            dataframe (pd.DataFrame): DataFrame containing simulation and reference values.

        Returns:
            float: Weighted MSE value.
        """
        # TODO: Normalizing and applying weights based on emission contribution might be a better approach.

        # Drop rows with NaN values in key columns
        filtered_df = dataframe.dropna(subset=['Simulation_Values', 'Edgar_Values', 'diff'])

        # Filter rows with valid EDGAR values to avoid infinite weights
        filtered_df = filtered_df[filtered_df['Edgar_Values'] != 0]

        # Ensure `diff` is absolute for weights
        filtered_df['weight'] = filtered_df['diff'].abs()

        # Calculate squared differences between Simulation_Values and Edgar_Values
        filtered_df['squared_error'] = (filtered_df['Simulation_Values'] - filtered_df['Edgar_Values']) ** 2

        # Calculate Weighted MSE
        if filtered_df['weight'].sum() == 0:  # Avoid division by zero
            return float('inf')  # Assign a high error if weights are invalid

        weighted_mse_value = (filtered_df['squared_error'] * filtered_df['weight']).sum() / filtered_df['weight'].sum()
        return weighted_mse_value
    

    def rmse(self, dataframe):
        """
        Computes the Root Mean Squared Error (RMSE).

        Args:
            dataframe (pd.DataFrame): DataFrame containing simulation and reference values.

        Returns:
            float: RMSE value.
        """
        # Drop rows with NaN values in the relevant columns
        filtered_df = dataframe.dropna(subset=['Simulation_Values', 'Edgar_Values'])

        # Filter rows with valid EDGAR values to avoid skewing the RMSE
        filtered_df = filtered_df[filtered_df['Edgar_Values'] != 0]

        # Calculate squared differences
        filtered_df['squared_error'] = (filtered_df['Simulation_Values'] - filtered_df['Edgar_Values']) ** 2

        # Compute RMSE
        rmse_value = (filtered_df['squared_error'].mean()) ** 0.5
        return rmse_value
    
    def mae(dataframe):
        """
        Computes Mean Absolute Error (MAE).

        Args:
            dataframe (pd.DataFrame): DataFrame with Simulation and Edgar values.

        Returns:
            float: MAE value.
        """
        # TODO: Complete this method
        
        # Drop rows with NaN in key columns
        filtered_df = dataframe.dropna(subset=['Simulation_Values', 'Edgar_Values'])
        
        # Compute MAE
        mae_value = (filtered_df['Simulation_Values'] - filtered_df['Edgar_Values']).abs().mean()
        return mae_value
    
    def calculate_error(self, error_type, dataframe):
        """
        Calculates the error based on the specified error type.

        Args:
            dataframe (pd.DataFrame): DataFrame containing simulation and reference values.

        Returns:
            float: Error value.
        """
        if error_type == 'weighted_mse':
            return self.weighted_mse(dataframe)
        elif error_type == 'rmse':
            return self.rmse(dataframe)
        elif error_type == 'mae':
            return self.mae(dataframe)
        else:
            raise ValueError(f"Error type '{error_type}' is not supported.")
