import pandas as pd
import numpy as np
import os

class DiffReportUtils:
    """
    DiffReportUtils is a utility class for handling and processing SSP and EDGAR emissions data. 
    It provides methods to load, clean, and transform data, calculate weights and deviations, 
    and generate emissions reports.
    Methods:
        __init__(self, iso_alpha_3, ssp_edgar_cw_path, sim_init_year=2015, comparison_year=2015):
            Initializes the DiffReportUtils class with the given parameters.
        load_ssp_edgar_cw(self):
        clean_ssp_out_data(self, ssp_out_df):
        calculate_weights(self, edgar_region_df):
        edgar_emission_db_etl(self, file_path):
        generate_ssp_emissions_report(self, ssp_out_df):
        adjust_duplicated_edgar_classes(self, df_ssp_edgar):
        calculate_ssp_edgar_deviation(self, df_ssp_edgar):
        merge_ssp_with_edgar(self, ssp_emissions_report, edgar_emissions_df):
    """
    
    def __init__(self, iso_alpha_3, ssp_edgar_cw_path, sectoral_report_dir_path, energy_model_flag, sim_init_year=2015, comparison_year=2015):
        """
        Initializes the DiffReports class with the given parameters.

        Args:
            iso_alpha_3 (str): The ISO 3166-1 alpha-3 region code.
            ssp_edgar_cw_path (str): The file path to the SSP EDGAR CW csv file.
            sim_init_year (int, optional): The initial year for the simulation. Defaults to 2015.
            comparison_year (int, optional): The year for comparison between SSP and EDGAR. Defaults to 2015.

        Attributes:
            ssp_edgar_cw_path (str): Stores the file path to the SSP EDGAR CW csv file.
            iso_alpha_3 (str): Stores the ISO 3166-1 alpha-3 region code.
            epsilon (float): A small constant (1e-6) to prevent numerical issues.
            model_failed_flag (bool): A flag indicating whether the model has failed. Defaults to False.
            sim_init_year (int): Stores the initial year for the simulation.
            comparison_year (int): Stores the year for comparison between SSP and EDGAR.
        """
        self.ssp_edgar_cw_path = ssp_edgar_cw_path
        self.sectoral_report_dir_path = sectoral_report_dir_path
        self.iso_alpha_3 = iso_alpha_3
        self.epsilon = 1e-6
        self.model_failed_flag = False
        self.energy_model_flag = energy_model_flag
        self.sim_init_year = sim_init_year
        self.comparison_year = comparison_year
        self.sectoral_emission_report = pd.DataFrame()
        self.subsector_emission_report = pd.DataFrame()

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

        # Remove blank spaces from column names
        ssp_edgar_cw.columns = ssp_edgar_cw.columns.str.strip()
        
        # Set column names to lowercase
        ssp_edgar_cw.columns = ssp_edgar_cw.columns.str.lower()

        # Filter by ignore column
        ssp_edgar_cw = ssp_edgar_cw[ssp_edgar_cw['ignore'] != 1]

        # Reset index
        ssp_edgar_cw = ssp_edgar_cw.reset_index(drop=True)
        
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
                - edgar_emission: Emission value for the comparison year.
                - year: The comparison year.
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
        edgar_region_df = edgar_region_df[['Code', 'edgar_class', str(self.comparison_year)]]

        # Rename columns
        edgar_region_df.rename(columns={str(self.comparison_year): 'edgar_emission', 'Code': 'iso_alpha_3'}, inplace=True)

        # Add a year column
        edgar_region_df['year'] = int(self.comparison_year)

        # Add weight columns NOTE: This will be uncommented when we have the complete mapping of Edgar classes
        # edgar_region_df = self.calculate_weights(edgar_region_df)

        # Create a edgar_emission_epsilon column NOTE: This will be uncommented when we have the complete mapping of Edgar classes
        # edgar_region_df['edgar_emission_epsilon'] = edgar_region_df['edgar_emission'] + self.epsilon
        
        return edgar_region_df
    
    def generate_ssp_emissions_report(self, ssp_out_df):
        """
        Generate an SSP emissions report based on the provided simulation DataFrame.
        This method creates a draft SSP emissions report from the `ssp_edgar_cw` table,
        calculates total emissions for each row based on the variables specified in the
        'vars' column, and updates the report with these values. It also flags if any
        variables specified in 'vars' are missing from the simulation DataFrame.
        Args:
            ssp_out_df (pd.DataFrame): A DataFrame containing simulation data with
                                          columns representing different variables.
        Returns:
            pd.DataFrame: A DataFrame containing the SSP emissions report with total
                          emissions calculated and the 'vars' column removed.
        Side Effects:
            - Sets `self.model_failed_flag` to True if any variables specified in 'vars'
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

        # Iterate through each row in detailed_report_draft_df
        for index, row in ssp_emissions_report.iterrows():
            
            vars_list = row['vars'].split(':') if pd.notna(row['vars']) else []  # Split Vars column into variable names if not NaN

            # Set missing_variables to True if any variable in vars_list is not in simulation_df otherwise False
            missing_variables = any(var not in simulation_df_cols for var in vars_list)

            #NOTE: DEBUG ONLY PLEASE REMOVE
            # print(f"Missing variables for {row['subsector']} {missing_variables}")
            
            if missing_variables:
                # Set subsector_total_emissions to NaN if missing variables
                subsector_total_emissions = np.nan
            elif vars_list:
                # Sum columns in the simulation df
                subsector_total_emissions = simulation_df[vars_list].sum(axis=1).values[0]
            else:
                # Set subsector_total_emissions to NaN if vars_list is empty
                subsector_total_emissions = np.nan

            # Update the simulation_values column in detailed_report_draft_df
            ssp_emissions_report.at[index, 'ssp_emission'] = subsector_total_emissions

        # Set model_failed_flag to True if there are missing variables
        if missing_variables and self.energy_model_flag:
            print(f"Missing variables for {row['subsector']}")
            self.model_failed_flag = True

        elif missing_variables and not self.energy_model_flag:
            
            if row['subsector'] not in ['entc', 'fgtv', 'ccsq']:
                print(f"Missing variables for {row['subsector']}")
                self.model_failed_flag = True
            else: 
                self.model_failed_flag = False
        else:
            self.model_failed_flag = False

        # Drop unnecessary columns
        ssp_emissions_report.drop(columns=['vars',
                                           'edgar_subsector',
                                           'edgar_sector',
                                           'ignore', 
                                           'note', 
                                           'need_better_information_on_what_is_contained'
                                           ], 
                                           inplace=True)

        return ssp_emissions_report
    
    def group_ssp_emissions_report_vars(self, ssp_emissions_report):
        """
        Groups SSP emissions report variables by subsector and EDGAR class.

        This function takes a DataFrame containing SSP emissions report data, renames specific subsectors,
        and then groups the data by subsector and EDGAR class, summing the SSP emissions for each group.

        Parameters:
        ssp_emissions_report (pd.DataFrame): A DataFrame containing SSP emissions report data with columns 
                                             'edgar_class', 'subsector', and 'ssp_emission'.

        Returns:
        pd.DataFrame: A DataFrame grouped by 'subsector' and 'edgar_class' with summed 'ssp_emission' values.
        """

        df = ssp_emissions_report.copy()

        # Rename the subsectors to group
        df.loc[df['edgar_class'] == "AG - Livestock:CH4", "subsector"] = "lvst-lsmm"

        # Group by subsector and edgar_class
        df = df.groupby(['subsector', 'edgar_class'], as_index=False)['ssp_emission'].sum()

        return df


    #NOTE: This method is a temporal fix
    def adjust_duplicated_edgar_classes(self, df_ssp_edgar):
        """
        Adjusts duplicated EDGAR classes in the given DataFrame by redistributing the emission values.
        This method identifies rows in the DataFrame that have duplicated 'edgar_class' values. For each duplicated class,
        it calculates a new emission value by dividing the original emission value by the number of duplicated rows. It then
        updates the 'edgar_emission' values of the duplicated rows with this new emission value.
        Parameters:
            df_ssp_edgar (pd.DataFrame): DataFrame containing the EDGAR data with 'edgar_class' and 'edgar_emission' columns.
            Returns:
            pd.DataFrame: A new DataFrame with adjusted 'edgar_emission' values for duplicated 'edgar_class' entries.
        """

        df = df_ssp_edgar.copy()

        # Get uplicated Edgar classes
        duplicated_classes = list(df[df.duplicated(subset=['edgar_class'], keep=False)]['edgar_class'].unique())

        # Iterate through duplicated classes
        for edgar_class in duplicated_classes:
            # Get duplicated rows
            duplicated_rows = df[df['edgar_class'] == edgar_class]
            
            # The duplicated rows have the same value in the edgar_emission column so we get the first value and divide it by the number of duplicated rows
            new_emission = duplicated_rows['edgar_emission'].iloc[0] / len(duplicated_rows)

            # print(f"Updating {edgar_class} with {new_emission}")

            # Update the edgar_emission values of the duplicated rows with new_emission value
            df.loc[df['edgar_class'] == edgar_class, 'edgar_emission'] = new_emission
        
        return df
    
    def calculate_ssp_edgar_deviation(self, df_ssp_edgar):
        """
        Calculate the deviation and squared deviation between SSP and EDGAR emissions.
        This function takes a DataFrame containing SSP and EDGAR emissions data, and calculates
        the deviation and squared deviation between the SSP emissions and EDGAR emissions epsilon.
        Parameters:
        df_ssp_edgar (pd.DataFrame): DataFrame containing 'ssp_emission' and 'edgar_emission_epsilon' columns.
        Returns:
        pd.DataFrame: A copy of the input DataFrame with additional columns:
            - 'rel_error': The deviation between 'ssp_emission' and 'edgar_emission_epsilon'.
            - 'squared_diff': The squared deviation between 'ssp_emission' and 'edgar_emission_epsilon'.
        """

        df = df_ssp_edgar.copy()
        df['rel_error'] = (df['ssp_emission'] - df['edgar_emission_epsilon']) / df['edgar_emission_epsilon']
        df['squared_diff'] = (df['edgar_emission_epsilon'] - df['ssp_emission']) ** 2
        return df

    
    def merge_ssp_with_edgar(self, ssp_emissions_report, edgar_emissions_df):
        """
        Merges the SSP emissions report with the EDGAR emissions dataframe, adjusts for duplicated EDGAR classes,
        calculates temporary epsilon-adjusted emissions, weights, and deviations, and handles missing data.
        Parameters:
            ssp_emissions_report (pd.DataFrame): The SSP emissions report dataframe.
            edgar_emissions_df (pd.DataFrame): The EDGAR emissions dataframe.
        Returns:
            pd.DataFrame: The merged dataframe with calculated deviations and handled missing data.
        """
        
        df = ssp_emissions_report.copy()

        # Merge the ssp_emissions_report with edgar_emissions_df
        df_merged = pd.merge(df, edgar_emissions_df, how='left', on='edgar_class')

        #NOTE: This is a temporal fix until we have a complete mapping of Edgar classes
        #NOTE: Commented out since we are now groupping the ssp subsectors to match edgar class
        #df_merged = self.adjust_duplicated_edgar_classes(df_merged)
    
        #Reset year to ref year to avoid NaNs
        df_merged['year'] = self.comparison_year

        return df_merged
    
    def group_trns_scoe_inen(self, ssp_edgar_df):

        df = ssp_edgar_df.copy()

        # Group the subsectors 'trns', 'inen', and 'scoe', and sum the 'ssp_emission' and 'edgar_emission' columns
        df_grouped = df[df['subsector'].isin(['trns', 'inen', 'scoe'])].groupby(['subsector', 'iso_alpha_3', 'year'], as_index=False).agg({
            'ssp_emission': 'sum',
            'edgar_emission': 'sum'
        })

        new_edgar_class = {
            'trns': 'EN - Transportation',
            'inen': 'EN - Manufacturing/Construction',
            'scoe': 'EN - Building'
        }

        # Add the new edgar_class column
        df_grouped['edgar_class'] = df_grouped['subsector'].map(new_edgar_class)

        # Reorder the columns
        df_grouped = df_grouped[['subsector', 'edgar_class', 'ssp_emission', 'iso_alpha_3', 'edgar_emission', 'year']]

    
        # Add the remaining rows that do not belong to the selected subsectors
        df_other = df[~df['subsector'].isin(['trns', 'inen', 'scoe'])]

        # Concatenate the grouped dataframe with the remaining rows
        df_final = pd.concat([df_grouped, df_other], ignore_index=True)

        return df_final
    
    
    def generate_subsector_diff_report(self, detailed_diff_report_complete):
        """
        Generates a subsector difference report by comparing simulation values with Edgar values.
        Args:
            detailed_diff_report_complete (pd.DataFrame): DataFrame containing detailed difference report with columns 
                                                          'Subsector', 'Simulation_Values', and 'Edgar_Values'.
        Returns:
            pd.DataFrame: A DataFrame containing the subsector difference report with columns 'Year', 'Subsector', 
                          'Simulation_Values', 'Edgar_Values', and 'rel_error'. The 'rel_error' column represents the difference 
                          between 'Simulation_Values' and 'Edgar_Values' as a fraction of 'Edgar_Values'.
        """
        
        # Group by Subsector and calculate the sum of the Simulation_Values and Edgar_Values
        subsector_diff_report = detailed_diff_report_complete.groupby('subsector')[['ssp_emission', 'edgar_emission_epsilon']].sum().reset_index()

        # Replace 0 values in ssp_emission with NaNs
        subsector_diff_report['ssp_emission'] = subsector_diff_report['ssp_emission'].replace(0, np.nan)

        # Calculate diffs
        subsector_diff_report = self.calculate_ssp_edgar_deviation(subsector_diff_report)
 
        # Reset Year column to ref year to avoid NaN values
        subsector_diff_report['year'] = self.comparison_year

        return subsector_diff_report
    
    
    
    def run_report_generator(self, edgar_emission_df, ssp_out_df, subsector_to_calibrate=None):
        """
        Run the report generator to generate a sectoral emissions report.
        This method generates a sectoral emissions report by merging the SSP emissions report with the EDGAR emissions data,
        calculating deviations and squared deviations, and handling missing data. It then sets the `sectoral_emission_report`
        attribute to the generated report.
        Args:
            edgar_emission_df (pd.DataFrame): The EDGAR emissions data DataFrame.
            ssp_out_df (pd.DataFrame): The SSP output data DataFrame.
        Returns:
            pd.DataFrame: The sectoral emissions report DataFrame.
        """
        
        # Generate the SSP emissions report
        ssp_emissions_report = self.generate_ssp_emissions_report(ssp_out_df)

        # Group the ssp_emissions_report to match the edgar_emission_df. NOTE: This is a temporal fix
        ssp_emissions_report = self.group_ssp_emissions_report_vars(ssp_emissions_report)

        # Merge the SSP emissions report with the EDGAR emissions data
        merged_df = self.merge_ssp_with_edgar(ssp_emissions_report, edgar_emission_df)

        # Aggregate trns, scoe and inen #NOTE: This is in testing phase
        merged_df = self.group_trns_scoe_inen(merged_df)

        # If energy_model_flag is False remove the rows with 'fgtv', 'entc', and 'ccsq' in subsector column
        if not self.energy_model_flag:
            merged_df = merged_df[~merged_df['subsector'].isin(['fgtv', 'entc', 'ccsq'])]

        # Drop rows with NaNs in edgar_emission columns. NOTE: This might not be needed when the mapping is complete
        merged_df = merged_df.dropna(subset=['edgar_emission'])


        #NOTE: We create edgar_emission_epsilon here temporarily until we have the complete mapping of Edgar classes
        merged_df['edgar_emission_epsilon'] = merged_df['edgar_emission'] + self.epsilon

        #Calculate rel_error and squared error
        merged_df = self.calculate_ssp_edgar_deviation(merged_df)

        # After all filtering we calculate the weights
        merged_df = self.calculate_weights(merged_df)

        # Generate subsector emission report
        subsector_diff_report = self.generate_subsector_diff_report(merged_df)

        # Filter the merged_df and subsector_diff_report for the subsector to calibrate

        if subsector_to_calibrate is not None:
            merged_df = merged_df[merged_df['subsector'] == subsector_to_calibrate]
            subsector_diff_report = subsector_diff_report[subsector_diff_report['subsector'] == subsector_to_calibrate]

        
        # Set the sectoral_emission_report attribute to the generated report
        self.sectoral_emission_report = merged_df.copy()
       
        # Set the subsector_emission_report attribute to the generated report
        self.subsector_emission_report = subsector_diff_report.copy()
        
        return None
    
    

    
    