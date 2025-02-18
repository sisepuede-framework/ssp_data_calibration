import pandas as pd
import numpy as np


class DiffReportUtils:
    
    def __init__(self, iso_alpha_3, ref_year, ssp_edgar_cw_path, sim_init_year=2015, comparison_year=2015):
        self.ssp_edgar_cw_path = ssp_edgar_cw_path
        self.iso_alpha_3 = iso_alpha_3
        self.ref_year = ref_year
        self.epsilon = 1e-6
        self.model_failed_flag = False
        self.sim_init_year = sim_init_year
        self.comparison_year = comparison_year

    def load_ssp_edgar_cw(self):
        """
        Load the SSP Edgar CW data from a CSV file.
        This method reads the SSP Edgar CW data from the CSV file specified by 
        the `ssp_edgar_cw_path` attribute and returns it as a pandas DataFrame.
        Returns:
            pd.DataFrame: The SSP Edgar CW data loaded from the CSV file.
        """
        # Load cw tables
        ssp_edgar_cw = pd.read_csv(self.ssp_edgar_cw_path)
        
        return ssp_edgar_cw
    
    def clean_ssp_out_data(self, ssp_out_df):
        """
        Cleans the SSP output data by adding a year column and filtering for the comparison year.
        Args:
            ssp_out_df (pd.DataFrame): The SSP output data frame to be cleaned.
        Returns:
            pd.DataFrame: The cleaned SSP output data frame containing only the data for the comparison year.
        Raises:
            ValueError: If the comparison year is not present in the simulation data.
        """
        
        ssp_out_clean = ssp_out_df.copy()

        # Add a year column to the df
        ssp_out_clean['year'] = ssp_out_clean['time_period'] + self.sim_init_year

        # Check if comparison_year is present in the simulation data
        if self.comparison_year not in ssp_out_clean['year'].unique():
            raise ValueError(f"The comparison year {self.comparison_year} is not present in the simulation data. Please increase your range of simulated years.")

        # Get only the data for the comparison year
        ssp_out_clean = ssp_out_clean[ssp_out_clean['year'] == self.comparison_year]
 
        return ssp_out_clean
    
    
    def calculate_weights(self, edgar_region_df):
        """
        Calculate different types of weights based on the 'edgar_emission' column of the input DataFrame.
        Parameters:
        edgar_region_df (pd.DataFrame): DataFrame containing 'edgar_emission' column with emission data.
        Returns:
        pd.DataFrame: DataFrame with additional columns for direct weights, normalized weights, and log weights.
            - 'direct_weight': Absolute value of 'edgar_emission' plus a small constant (epsilon).
            - 'norm_weight': Normalized weights calculated as the absolute value of 'edgar_emission' divided by the sum of absolute values of 'edgar_emission'.
            - 'log_weight': Logarithmic weights calculated as the natural logarithm of the absolute value of 'edgar_emission' plus 1.
        """
    
        df = edgar_region_df.copy()
        
        # Calculate direct weights
        df['direct_weight'] = np.abs(df['edgar_emission']) + self.epsilon
        
        # Calculate normalized weights
        df['norm_weight'] = df['edgar_emission'].abs() / df['edgar_emission'].abs().sum()

        # Calculate log weights
        df['log_weight'] = np.log(np.abs(df["edgar_emission"]) + 1)

        return df
    
    
    def edgar_emission_db_etl(self, file_path):
        """
        Extract, transform, and load (ETL) process for EDGAR emissions database.
        This method reads a CSV file containing EDGAR emissions data, filters it for a specific region,
        processes the data to create relevant columns, and calculates weights and adjusted emissions.
        Args:
            file_path (str): The file path to the EDGAR emissions database CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the processed EDGAR emissions data with the following columns:
                - iso_alpha_3: ISO alpha-3 region code.
                - edgar_class: Combined class of CSC Subsector and Gas.
                - edgar_emission: Emission value for the reference year.
                - year: The reference year.
                - weight columns: Additional columns calculated by the `calculate_weights` method.
                - edgar_emission_epsilon: Adjusted emission value with epsilon added.
        """
       
        # Load db csv
        edgar_emissions_db = pd.read_csv(file_path, encoding='latin1')

        # Filter the db for the specific region
        edgar_region_df = edgar_emissions_db[edgar_emissions_db['Code'] == self.iso_alpha_3].reset_index(drop=True)

        # Create edgar_class column by combining Subsector and Gas columns
        edgar_region_df['edgar_class'] = edgar_region_df['CSC Subsector'] + ':' + edgar_region_df['Gas']

        # Keep relevant columns
        edgar_region_df = edgar_region_df[['Code', 'edgar_class', str(self.ref_year)]]

        # Rename columns
        edgar_region_df.rename(columns={str(self.ref_year): 'edgar_emission', 'Code': 'iso_alpha_3'}, inplace=True)

        # Add a year column
        edgar_region_df['year'] = int(self.ref_year)

        # Add weight columns
        edgar_region_df = self.calculate_weights(edgar_region_df)

        # Create a edgar_emission_epsilon column
        edgar_region_df['edgar_emission_epsilon'] = edgar_region_df['edgar_emission'] + self.epsilon
        
        return edgar_region_df
    
    def generate_ssp_emissions_report(self, ssp_out_df):
        """
        Generate an SSP emissions report based on the provided simulation DataFrame.
        This method creates a draft SSP emissions report from the `ssp_edgar_cw` table,
        calculates total emissions for each row based on the variables specified in the
        'Vars' column, and updates the report with these values. It also flags if any
        variables specified in 'Vars' are missing from the simulation DataFrame.
        Args:
            ssp_out_df (pd.DataFrame): A DataFrame containing simulation data with
                                          columns representing different variables.
        Returns:
            pd.DataFrame: A DataFrame containing the SSP emissions report with total
                          emissions calculated and the 'Vars' column removed.
        Side Effects:
            - Sets `self.model_failed_flag` to True if any variables specified in 'Vars'
              are missing from the simulation DataFrame, otherwise sets it to False.
        """
        
        # Clean the simulation data
        simulation_df = self.clean_ssp_out_data(ssp_out_df)
        
        # Create a draft for ssp_emissions_report from the ssp_edgar_cw table
        ssp_emissions_report = self.load_ssp_edgar_cw()
        
        # Add a new column to store total emissions
        ssp_emissions_report['ssp_emission'] = 0.0  # Initialize with float zeros

        # Create a set of all column names in simulation_df for quick lookup
        simulation_df_cols = set(simulation_df.columns)

        # Create a set to store all missing variable names
        missing_variables = set()

        # Iterate through each row in detailed_report_draft_df
        for index, row in ssp_emissions_report.iterrows():
            vars_list = row['Vars'].split(':')  # Split Vars column into variable names

            # Check which variable names are missing in simulation_df
            missing_in_row = [var for var in vars_list if var not in simulation_df_cols]
            missing_variables.update(missing_in_row)  # Add missing variables to the set

            # Filter the columns in simulation_df that match the variable names
            matching_columns = [col for col in vars_list if col in simulation_df_cols]

            if matching_columns:
                # Sum the matching columns to get the total emissions for the subsector
                subsector_total_emissions = simulation_df[matching_columns].sum(axis=1).values[0]
            else:
                subsector_total_emissions = -999  # No matching columns found

            # Update the simulation_values column in detailed_report_draft_df
            ssp_emissions_report.at[index, 'ssp_emission'] = subsector_total_emissions

        # Set model_failed_flag to True if there are missing variables
        if missing_variables:
            # print("The following variables from Vars are not present in simulation_df:")
            # for var in missing_variables:
            #     print(var)
            self.model_failed_flag = True
        else:
            self.model_failed_flag = False

        # Drop the Vars column
        ssp_emissions_report.drop(columns=['Vars'], inplace=True)

        # Make column names lowercase
        ssp_emissions_report.columns = ssp_emissions_report.columns.str.lower()

        return ssp_emissions_report
    

    
    