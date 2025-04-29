import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iapws import IAPWS97
from utils.default_input_class_GTEcon import well_inputs, econ_inputs
from utils.hydraulic_utils import hydraulic_loss_2
import sys
from tqdm import tqdm
import seaborn as sns
import os
from CoolProp.CoolProp import PropsSI
import warnings

@pd.api.extensions.register_dataframe_accessor("gte")
class GTEcon:
    def __init__(
        self,
        well_operation_data_obj: pd.DataFrame = None,
        well_operation_path: str = None,
        params_path: str = None):
        """
        Initialize the GTEcon model by loading well operation data and optional parameter file.

        Parameters:
            well_operation_data_obj (pd.DataFrame, optional): DataFrame containing well operation time series data.
                If not provided, well_operation_path must be specified.
            well_operation_path (str, optional): Path to Excel file with time series data.
                Required if well_operation_data_obj is None.
            params_path (str, optional): Path to Excel file with sheets:
                'Units', 'Well_Inputs', 'Cost_Inputs', and 'Econ_Inputs'.
                If provided, parameters will be loaded automatically via _load_params().
                Otherwise, user must call:
                  - assign_units(df_units)
                  - assign_well_inputs(df_well)
                  - assign_cost_inputs(df_cost)
                  - assign_econ_inputs(df_econ)
                  before running the model.
        """
        # --- Load or assign time series data ---
        if well_operation_data_obj is None:
            if well_operation_path:
                try:
                    data = pd.read_excel(well_operation_path)
                except Exception as e:
                    raise ValueError(f"Error reading data file '{well_operation_path}': {e}")
                self.initial_data = data
                # self._obj = data.copy()
            else:
                raise ValueError(
                    "Either 'well_operation_data_obj' or 'well_operation_path' must be provided."
                )
        else:
            self.initial_data = well_operation_data_obj.copy()
            # self._obj = well_operation_data_obj.copy()

        # --- Initialize attributes ---
        self._obj=pd.DataFrame()
        self.verbose = False
        self.warning = True
        self.is_plotting = False
        self.is_first_run = True
        self.interpolated_dates_df = None

        # Initialize input holders
        self.econ_inputs = econ_inputs()
        self.well_inputs = well_inputs()

        # Result placeholders
        self.npv = None
        self.lcoe = None
        self.payback_period_cf = None  # Payback time (years) based on the cumulative cash flow
        self.payback_period_npv = None  # Payback time (years) based on the NPV:

        # --- Load parameters if provided ---
        if params_path:
            try:
                self._load_params(params_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to load parameters from '{params_path}': {e}"
                )
        else:
            warnings.warn(
                "No parameter file provided. Please call the following methods before running the model:\n"
                "  – assign_units(df_units)\n"
                "  – assign_well_inputs(df_well)\n"
                "  – assign_cost_inputs(df_cost)\n"
                "  – assign_econ_inputs(df_econ)\n",
                UserWarning
            )

    def _load_params(self, params_path: str):
        """
        Load Units, Well_Inputs, Cost_Inputs, and Econ_Inputs from an Excel file.

        Parameters:
            params_path (str): Path to Excel file with sheets:
                'Units', 'Well_Inputs', 'Cost_Inputs', 'Econ_Inputs'.
        """
        # Suppress OpenPyXL “Data Validation extension is not supported” warning when reading Excel files
        warnings.filterwarnings(
            "ignore",
            message="Data Validation extension is not supported and will be removed")
        
        df_units = pd.read_excel(params_path, sheet_name='Units')
        df_well = pd.read_excel(params_path, sheet_name='Well_Inputs')
        df_cost = pd.read_excel(params_path, sheet_name='Cost_Inputs')
        df_econ = pd.read_excel(params_path, sheet_name='Econ_Inputs')

        # Assign loaded data
        self.assign_units(df_units)
        self.assign_well_inputs(df_well)
        self.assign_cost_inputs(df_cost)
        self.assign_econ_inputs(df_econ)

    def assign_well_inputs(self, df_well_inputs):
        try:
            self.well_inputs.well_names = df_well_inputs['well_names'].tolist()
            self.well_inputs.well_types = df_well_inputs['well_types'].tolist()
            self.well_inputs.well_depths_m = df_well_inputs['well_depths_m'].tolist()
            self.well_inputs.reservoir_depth_tvd_m = df_well_inputs['reservoir_depth_tvd_m'].tolist()
            self.well_inputs.pump_efficiency = df_well_inputs['pump_efficiency'].tolist()
            self.well_inputs.T_column_names = df_well_inputs['T_column_names'].tolist()
            self.well_inputs.P_column_names = df_well_inputs['P_column_names'].tolist()
            self.well_inputs.Q_column_names = df_well_inputs['Q_column_names'].tolist()
        except Exception as e:
            print(f"Error: Well input parameters can not be assigned! Check the input file!\nDetails: {e}")
            sys.exit()

    def assign_econ_inputs(self, df_econ_inputs):
        try:
            # self.econ_inputs.unique_well_cost = df_econ_inputs.iloc[0]['unique_well_cost']
            # self.econ_inputs.calculate_well_cost = df_econ_inputs.iloc[0]['calculate_well_cost']
            # self.econ_inputs.OpEx_annual_percent_CapEx = df_econ_inputs.iloc[0]['OpEx_annual_percent_CapEx']
            self.econ_inputs.annual_discount_rate = df_econ_inputs.iloc[0]['annual_discount_rate']
            self.econ_inputs.heat_price_euro_MWh = df_econ_inputs.iloc[0]['heat_price_euro_MWh']
            self.econ_inputs.electricity_price_euro_MWh = df_econ_inputs.iloc[0]['electricity_price_euro_MWh']
            # self.econ_inputs.AbEx_per_well = df_econ_inputs.iloc[0]['AbEx_per_well']
            # self.econ_inputs.surface_piping_cost_euro_m = df_econ_inputs.iloc[0]['surface_piping_cost_euro_m']
        except Exception as e:
            print(f"Error: Economic input parameters can not be assigned! Check the input file!\nDetails: {e}")
            sys.exit()

    def assign_cost_inputs(self, df_cost_inputs):
        try:
            cost_comp = {}
            for _, row in df_cost_inputs.iterrows():
                component = row['Component']
                cost_comp[component] = {
                    'unit_cost': row['Unit Cost (€)'],
                    'quantity': row['Quantity'],
                    'replacement_freq': row['Replacement Interval (in year)']
                }
            self.econ_inputs.cost_comp = cost_comp
        except Exception as e:
            print(f"Error: Cost components can not be assigned! Check the input file!\nDetails: {e}")
            sys.exit()

    def assign_units(self, df_units):
        try:
            self.well_inputs.time_column_name = df_units.iloc[0]['Time_Column_Caption']
            self.well_inputs.CoEff_units = units(time=df_units.iloc[0]['Time_Unit'],
                                                 T=df_units.iloc[0]['Temp_Unit'],
                                                 P=df_units.iloc[0]['P_Unit'],
                                                 Q=df_units.iloc[0]['Q_Unit'])
        except Exception as e:
            print(f"Error: Units can not be assigned! Check the input file!\nDetails: {e}")
            sys.exit()

    def check_dt_hotwells(self):
        """
        Checks for temperature inconsistencies between hot wells and warm wells.
        Printing details if any hot well is cooler than a warm well.
        """
        hotwell_temp_col_names = self._obj.filter(like='_H_T').columns.tolist()
        warmwell_temp_col_names = self._obj.filter(like='_W_T').columns.tolist()

        for h in hotwell_temp_col_names:
            for w in warmwell_temp_col_names:
                dt = self._obj[h] - self._obj[w]
                if min(dt[1:]) < 0:
                    print(f"\n***** Temperature inconsistancy detected between Hot & Cold wells *****")
                    print(f"*** Temperature in {h} lower than {w}! Time details are given below ***\n")
                    print(dt[dt < 0].index)
                    print(
                        f"\n***The techno-economic module will still carry out the calculation,\n but be careful when interpreting the results!")
                    print(f"-------------------------------------------------------------------------")
                else:
                    print(f"\n***** Temperatures between Hot and Warm wells are consistant!*****")

    def check_well_flowrates(self):
        """
        Checks for flowrate inconsistencies between hot wells and warm wells.
        Printing warning if total flowrate of the hotwells is not equal to the warmwells.
        """
        # Identify columns containing '_H_T' and '_W_T'
        hotwell_flowrate_col_names = self._obj.filter(like='_H_Q').columns.tolist()
        warmwell_flowrate_col_names = self._obj.filter(like='_W_Q').columns.tolist()

        # Sum the values in the identified columns
        df = pd.DataFrame()
        df.index = self._obj.index.copy()
        df['hotwell total flow'] = self._obj[hotwell_flowrate_col_names].sum(axis=1)
        df['warmwell total flow'] = self._obj[warmwell_flowrate_col_names].sum(axis=1)

        # Check each row for discrepancies
        for index, row in df.iterrows():
            Q_H = abs(int(row['hotwell total flow']))
            Q_W = abs(int(row['warmwell total flow']))
            if Q_H != Q_W:
                print(f"\nWarning: At index {index}, the total hotwell flowrate ({Q_H}) "
                      f"does not match the total warmwell flowrate ({Q_W}).")

    def control_and_write_inputs(self):
        print("\n*******************Input Data Summmary*******************\n")
        print("-------------Well Inputs-------------\n")
        for attr, val in vars(self.well_inputs).items():
            print(f"{attr}: {val}")

        print("\n-------------Economic Inputs-------------\n")
        for attr, val in vars(self.econ_inputs).items():
            print(f"{attr}: {val}")

    def pump_power(self):
        """
        Compute the pumping power required based on reference pressure for each operated well
        """
        
        """
        ------------------Revisit the dp_pump calculation to implement the occasions below!------------------
         1) Surface pipeline pressure should be considered for pump power calculation at production stages!
         2) reservoir pressure head at producer may overcome well water column and pipeline pressure!
         3) dp_pump may exceed re-injection pump capacity. Calculation can be proceed but a warning message can be shown to the user!
          """
        
        m3_day_m3_sec = 1 / (24 * 60 * 60)

        # Calculate the required pumping power for the system
        # if the inputs are in MPa the result is also in MW (1e6 factor)
        for i, well in enumerate(self.well_inputs.well_names):
            well_label = well + '_' + self.well_inputs.well_types[i]
            P = self._obj[well_label + '_P']                               # Reservoir pressure at TVD (bar-a)
            h = self.well_inputs.reservoir_depth_tvd_m[i]                  # Water height (m)
            P_pl = self.well_inputs.pipeline_pressure                      # Surface pipeline pressure in operation (bar-a)
            # needed_pressure_in_res = well_water_column_pressure + self.well_inputs.pipeline_pressure  # bar-a 

            # Calculate del_P due to friction loss
            flw = self._obj[well_label + '_Q']
            T = self._obj[well_label + '_T']
            L = self.well_inputs.well_depths_m[i]
            D = 0.15                                 # m, pipe inner diameter
            epsilon = 3e-6                           # m, pipe roughness
            
            
            #--------------------------------------------------------
            dp_pump_list = []
            
            # Iterate over each pair of flow, temperature and pressure
            for flow_m3_day, T_celsius, P_res in zip(flw, T, P):
                # Convert temperature to Kelvin
                T_K = T_celsius + 273.15
                # Convert pressure from bar to Pa (1 bar = 1e5 Pa)
                P_Pa = P_res * 1e5
                
                # Calculate water properties using CoolProp
                rho = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water')
                mu = PropsSI('VISCOSITY', 'T', T_K, 'P', P_Pa, 'Water')
                
                P_wc=(rho*9.81*h)/1e5       #Borehole water column pressure (bar-a)
                
                if flow_m3_day==0:           #If there is no flow, friction loss will be zero
                    dp_pump_list.append(0)
                else:
                    P_fric = hydraulic_loss_2(flow_m3_day, rho, D, mu, epsilon, L)
                    
                    if flow_m3_day>0:    #positive flow rate means injection
                        dp_pump = P_res + P_fric - P_wc
                        
                        if dp_pump<0:
                            #pump power is not needed when P_wc higher than (P_res[i] + P_fric).
                            #In this case the well can suck the water itself!
                            dp_pump_list.append(0)
                        else:
                            dp_pump_list.append(dp_pump)
                    
                    elif flow_m3_day<0:  #negative flow rate means production
                        dp_pump = P_wc + P_fric + P_pl - P_res
                        
                        if dp_pump<0:
                            #pump power is not needed when P_res higher than (P_wc+P_fric+P_pl).
                            dp_pump_list.append(0)
                        else:
                            dp_pump_list.append(dp_pump)
            #--------------------------------------------------------

            
            self._obj['%s : Pump dp (MPa)' % well] = np.array(dp_pump_list) * 0.1  # conversion from bar to MPa
            self._obj['%s : Pump power (MW)' % well] = abs(self._obj['%s : Pump dp (MPa)' % well]) * \
                                                       abs(self._obj[well_label + '_Q'] * m3_day_m3_sec /
                                                           self.well_inputs.pump_efficiency[i])

        self._obj['Pump power total (MW)'] = self._obj[self._obj.filter(like=': Pump power (MW)').columns.tolist()].sum(
            axis=1)

    def power(self, isEnergyColumnExist=False):
        """
        Calculate the power of the system
        :return: power in MW
        """

        if isEnergyColumnExist:
            self.calculate_heat_power()
        else:
            for i, w in enumerate(self.well_inputs.well_names):
                well = w + '_' + self.well_inputs.well_types[i]
                self._obj[well + ': Heat Power (MW)'] = self._obj.apply(
                    lambda row: calculate_heatpower_columns(row, well), axis=1)

            self.calculate_heat_power()

        self._obj['Power net (MW)'] = self._obj['Heat Production Power (MW)'] - self._obj['Pump power total (MW)']

        # assign initial Power to nan
        self._obj.loc[self._obj.index == self._obj.index[0], 'Power net (MW)'] = np.nan

        # Compute the COP
        self._obj['COP (-)'] = self._obj['Heat Production Power (MW)'] / self._obj['Pump power total (MW)']

        # Compute Produced Energy at each time step
        self._obj['Produced Energy (MWh)'] = self._obj['Heat Production Power (MW)'] * self._obj['Deltahours']

        # Compute the cumulative produced energy
        self._obj['Produced Energy Cum. (TWh)'] = (self._obj['Produced Energy (MWh)'].cumsum()) * 1e-6

    def calculate_heat_power(self):

        hotwell_energy_col_names = self._obj.filter(like='_H: Heat Power (MW)').columns.tolist()
        warmwell_energy_col_names = self._obj.filter(like='_W: Heat Power (MW)').columns.tolist()

        self._obj['hotwell total heat power'] = self._obj[hotwell_energy_col_names].sum(axis=1)
        self._obj['warmwell total heat power'] = self._obj[warmwell_energy_col_names].sum(axis=1)

        self._obj['Heat Charge Power (MW)'] = self._obj.apply(
            lambda row: abs(row['hotwell total heat power']) - abs(row['warmwell total heat power'])
            if row['hotwell total heat power'] > 0 else 0, axis=1)

        self._obj['Heat Production Power (MW)'] = self._obj.apply(
            lambda row: abs(row['hotwell total heat power']) - abs(row['warmwell total heat power'])
            if row['hotwell total heat power'] < 0 else 0, axis=1)

    def calc_econ_params(self,
                         use_learning_curve=False,
                         use_drill_depth_calc=False,
                         drop_npv=False,
                         verbose=False,
                         warning=False,
                         samples=False,
                         **kwargs):
        """
        :param wells_depths: list of floats
            well depths to be used for the computation of drilling costs
        :param electricity_price: float
            price of electricity to run the pumps in Euro/MWh
        :param pump_replace: float
            frequency of replacing pumps in years
        :param pump_cost: float
            cost of purchasing each pump in euros
        :param heat_price: float
            price for produced heat in Euro/MWh
        :return:

        """

        # Compute the generated income
        self._obj['Income (\u20ac)'] = self._obj['Produced Energy (MWh)'] * \
                                       self.econ_inputs.heat_price_euro_MWh

        # assign initial CapEx
        self._obj['CapEx (\u20ac)'] = 0

        # Convert the 'CapEx (€)' column to float
        # if we don't do this conversion, compiler gives error for future pandas version
        self._obj['CapEx (\u20ac)'] = self._obj['CapEx (\u20ac)'].astype(float)

        for comp, comp_info in self.econ_inputs.cost_comp.items():
            if self.verbose:
                print('\n\t\tComponent:%s \n\t\tcost (€): %s \n\t\trecurring every %s years' % (
                    comp,
                    comp_info["unit_cost"] * comp_info["quantity"],
                    round(comp_info["replacement_freq"], 2)))

            self.recurring_costs(recurring_cost=comp_info["unit_cost"] * comp_info["quantity"],
                                 recurring_frequency=comp_info["replacement_freq"])

        if self.econ_inputs.AbEx_per_well:
            self._obj.loc[
                self._obj.index == self._obj.index[-1], 'CapEx (\u20ac)'] = self.econ_inputs.AbEx_per_well * \
                                                                            len(self.well_inputs.well_depths_m)

        # calculate OpEx costs for pumping
        self._obj['OpEx_pump (\u20ac)'] = self._obj['Pump power total (MW)'] * \
                                          self._obj['Deltahours'] * \
                                          self.econ_inputs.electricity_price_euro_MWh


        self._obj['CF (\u20ac)'] = - self._obj['CapEx (\u20ac)'].fillna(0) \
                                   - self._obj['OpEx_pump (\u20ac)'].fillna(0) \
                                   + self._obj['Income (\u20ac)'].fillna(0)

        # Compute the cumulative (undiscounted) cash flow
        self._obj['CF_cum'] = self._obj['CF (€)'].cumsum()

        #-----------------calculate NPV-----------------
        self._obj[r'NPV (€)'] = (self._obj['CF (\u20ac)'] / self._obj['disc_rate']).cumsum()

        NPVmulti = '1e-6'
        self._obj[r'NPV (€) $\times$ $10^%s$' % NPVmulti[-1]] = self._obj[r'NPV (€)'] * float(NPVmulti)

        if drop_npv:
            self._obj.drop(columns=[r'NPV (€) $\times$ $10^%s$' % NPVmulti[-1]], inplace=True)

        #-----------------calculate payback periods-----------------
        # Calculate payback period based on the cumulative cash flow:
        self.payback_period_cf = self.calc_payback_period(df=self._obj, column="CF_cum", verbose=verbose)

        # Calculate payback period based on the NPV:
        self.payback_period_npv = self.calc_payback_period(df=self._obj, column="NPV (€)", verbose=verbose)

        #-----------------calculate LCOH-----------------
        self._obj['LCOH costs'] = self._obj['CapEx (\u20ac)'].fillna(0) \
                                  + self._obj['OpEx_pump (\u20ac)'].fillna(0)

        self._obj['discounted LCOH costs'] = (self._obj['LCOH costs'] / self._obj['disc_rate']).cumsum()

        self._obj['discounted  LCOH energy'] = (self._obj['Produced Energy (MWh)'] / self._obj['disc_rate']).cumsum()

        self._obj[r'LCOH (€/MWh)'] = self._obj['discounted LCOH costs'] / self._obj['discounted  LCOH energy']

        self._obj.head()  # print summary of the dataframe

        self.npv = self._obj[r'NPV (€)'].iloc[-1]
        self.lcoe = self._obj[r'LCOH (€/MWh)'].iloc[-1]

        if verbose:
            print("\n*******************LCOE*******************")
            print(f"LCOE (€/MWh): {self.lcoe:.2f} for {self._obj['Time (yrs)'].iloc[-1]:.2f} (yr) heat storage.\n\n")
        return

    def calc_payback_period(self, df, column, verbose=False, warning=True):
        """
        Calculate the payback period using weighted interpolation from the given DataFrame.

        This function expects the input DataFrame 'df' to include a "Time (yrs)" column and a column specified
        by 'column'. The function supports only two column names:
          - "CF_cum": to compute the undiscounted payback period (using cumulative cash flow),
          - "NPV (€)": to compute the discounted payback period (using Net Present Value).

        The function identifies the first time step at which the specified column becomes positive. If a
        transition from a negative value to a positive value is detected between two consecutive time steps,
        weighted linear interpolation (using np.interp) is performed to estimate the exact break-even time.
        If no break-even point is found, or if the break-even occurs at the very first time step, a warning is
        issued and the function returns None.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing the data.
            column (str): The name of the column to use for payback calculation; it must be either "CF_cum" or "NPV (€)".
            verbose (bool): If True, prints detailed diagnostic information.

        Returns:
            float or None: The estimated payback period in years, or None if no break-even point is found or if
                           break-even occurs at the first time step.
                           :param warning:
        """
        # Identify the first time step where the specified column becomes positive
        pos_indices = df.index[df[column] > 0]
        if len(pos_indices) == 0:
            if self.warning:
                print("\n***Warning: No payback period found. The cumulative cash flow or NPV never become positive.\n")
            return None

        # Get the integer location of the first positive cumulative value
        pos = df.index.get_loc(pos_indices[0])
        t_curr = df.loc[df.index[pos], "Time (yrs)"]
        value_curr = df.loc[df.index[pos], column]

        if pos > 0:
            t_prev = df.loc[df.index[pos - 1], "Time (yrs)"]
            value_prev = df.loc[df.index[pos - 1], column]
            # Perform weighted interpolation if a valid sign change occurs
            if value_prev < 0 and value_curr > 0:
                t_break_even = np.interp(0, [value_prev, value_curr], [t_prev, t_curr])
                payback_time = t_break_even
                if self.verbose:
                    print(f"\nPayback period : {t_curr:.2f} years at first positive value ({value_curr:.2f} €)")
                    print(f"Payback period : {t_prev:.2f} years at last negative value ({value_prev:.2f} €)")
                    print(f"Interpolated break-even (payback) period: {payback_time:.2f} years.\n")
                return t_break_even
            else:
                payback_time = t_curr
                if self.verbose:
                    print(f"\nPayback period (no interpolation needed to be performed): {payback_time:.2f} years.\n")
                return t_curr
        else:
            if self.warning:
                print(
                    "\n***Warning: Break-even occurs at the first time step! Payback time is set to None. Please check the input data!\n")
            return None

    def recurring_costs(self, recurring_cost=1000, recurring_frequency=2):  # , ,recurring_frequency='365D'):
        """
        Add CapEx costs to cash flow with a recurring interval

        recurring_cost: float
            component cost in euro

        recurring_frequency: float
            The interval of recurrence in years.
            This determines how often the cost will be added.
            - If `recurring_frequency=1`, the cost recurs every year.
            - If `recurring_frequency=0.5`, the cost recurs every 6 months (half a year).
            - If `recurring_frequency=2`, the cost recurs every 2 years.
            - If `recurring_frequency=0`, the cost is added only once at the beginning (CAPEX).

        Returns
        -------
        pandas.Series
            A series representing the costs added to the cash flow over time, in euros.

        Notes
        -----
            - The function adds the recurring costs based on the specified frequency, aligning
              them with the time intervals present in the data frame.
            - If `recurring_frequency` is set to 0, the cost is treated as a one-time expense
              and added only at the first time index.
        """
        df_interval_time = pd.Timedelta(self._obj.Deltahours.mode()[0], unit='h')  # day
        recurring_freq_time = pd.Timedelta(365.25 * recurring_frequency, unit='d')  # day

        # If recurring, assign the recurring costs
        if recurring_frequency != 0:

            # Determine recurrence_index_time 
            # check if dataframe intervals is higher than recurring frequency
            if df_interval_time > recurring_freq_time:

                # find the nearest time position in days for the recurring costs
                recurrence_index_time = self._obj.index.get_indexer([df_interval_time], method='nearest')[0]

                # provide multiplication factor for recurring costs based on recurring intervals in dataframe intervals
                recurring_cost *= df_interval_time / recurring_freq_time

            # check if dataframe intervals is lower than recurring frequency
            else:
                # find the nearest time position in days for the recurring costs
                recurrence_index_time = self._obj.index.get_indexer([recurring_freq_time], method='nearest')[0]

            self._obj.loc[::recurrence_index_time, 'CapEx (\u20ac)'] += recurring_cost

        else:
            # self._obj['CapEx (\u20ac)'].iloc[0] += recurring_cost
            self._obj.loc[self._obj.index[0], 'CapEx (\u20ac)'] += recurring_cost

    def run(self, plotting=False, verbose=False, warning=False):
        """
        Run the Techno-Economic Assessment Module in the calculation workflow below.
        returns:lcoe,fig_input,fig_output
        """
        self.verbose = verbose
        self.warning = warning
        self.is_plotting = plotting

        if self.is_first_run:
            self.organize_input_data()
            self.interpolated_dates_df = self.check_exact_years()
            self.is_first_run = False

            if self.verbose:
                print("The GTEcon model was initialized.")

        if self.warning:
            self.check_dt_hotwells()
            self.check_well_flowrates()

        self.assign_econ_periods()
        self.calc_per_discount_rate(mode=2)
        self.calculate_deltahours()
        self.pump_power()
        self.power()
        self.calc_econ_params(verbose=verbose)
        self._obj.to_excel("GTEcon_results.xlsx")

        if self.verbose:
            self.control_and_write_inputs()

        if self.is_plotting:
            self.overview_plot_input(self.interpolated_dates_df)
            self.overview_plot_output()
            
        print("\n****GTEcon run succesfully!****\n")

        return

    def mc_econ(self, params=None, iterations=100, reps=1000, verbose=False, warning=False):
        """
        Perform a Monte Carlo simulation to calculate the Net Present Value (NPV) and Levelized Cost of Heat (LCOH)
        based on varying economic parameters such as Discount Rate, Heat Price, and Electricity Price.

        Parameters:
        params (dict, optional): A dictionary containing the mean and standard deviation values for the parameters.
                                 Keys should be "Discount Rate", "Heat Price", and "Electricity Price".
                                 Prices are in Euro currency!
                                 Values should be tuples with (mean, std_dev).
                                 Default values are used if not provided.
        iterations (int, optional): The number of iterations to run in the simulation. Default is 1000.
        reps (int, optional): The number of repetitions to run for bootstrap convergence analysis. Default is 1000.
        verbose (bool, optional): If True, prints detailed output during the simulation. Default is False.

        Returns:
        list: A list of DataFrames, each containing the NPV and LCOH for each iteration.

        Example Usage:
        results = economic_model.mc_npv(
            params={"Heat Price": (60, 0.15)},
            iterations=1000,
            reps=1000
        )
        """
        self.warning=warning
        self.verbose=verbose
        
        print("\n-------Monte Carlo simulation is running...-------\n")

        if self.is_first_run:
            print('Error: The econ module must be run at least once before initiating the Monte Carlo simulation.')
            print('Please complete a forward econ run to proceed with the Monte Carlo simulation.')
            return


        # Set default values
        default_params = {"Discount Rate": (self.econ_inputs.annual_discount_rate, 0),
                          "Heat Price": (self.econ_inputs.heat_price_euro_MWh, 0),
                          "Electricity Price": (self.econ_inputs.electricity_price_euro_MWh, 0)}

        # Update default values with provided parameters
        if params:
            default_params.update(params)

        random_seed = np.random.randint(0, iterations)
        np.random.seed(random_seed)

        df = pd.DataFrame()
        df["Discount Rate"] = np.random.normal(default_params["Discount Rate"][0],
                                               scale=default_params["Discount Rate"][1],
                                               size=iterations)

        df["Heat Price (€)"] = np.random.normal(default_params["Heat Price"][0],
                                                scale=default_params["Heat Price"][1],
                                                size=iterations)
        df["Electricity Price (€)"] = np.random.normal(default_params["Electricity Price"][0],
                                                       scale=default_params["Electricity Price"][1],
                                                       size=iterations)

        df["Index"] = df.index  # added to use index as hue in sns graph

        mc_list = []
        df_lasts = []

        for i in tqdm(range(iterations)):
            self.econ_inputs.annual_discount_rate = df.iloc[i]["Discount Rate"]
            self.econ_inputs.heat_price_euro_MWh = df.iloc[i]["Heat Price (€)"]
            self.econ_inputs.electricity_price_euro_MWh = df.iloc[i]["Electricity Price (€)"]

            self.calc_econ_params(verbose=verbose)

            df2 = pd.DataFrame()
            df2['Time (yrs)'] = self._obj['Time (yrs)'].copy()
            df2['NPV (€)'] = self._obj[r'NPV (€)'].copy()
            df2['LCOH (€/MWh)'] = self._obj[r'LCOH (€/MWh)'].copy()
            df2['Iteration'] = i + 1

            mc_list.append(df2)
            df_lasts.append(df2.iloc[[-1]])

        last_df = pd.concat(df_lasts)
        npv_df = pd.concat(mc_list)
        npv_df.reset_index(inplace=True, drop=True)

        #--------------------Drawing MC Overview Plot--------------------
        fig, gs = plt.subplots(figsize=(12, 9), ncols=2, nrows=3, dpi=300)

        # Discount Rate Plot
        sns.scatterplot(x=df.index, y='Discount Rate', data=df, ax=gs[0, 0], legend=False)
        gs[0, 0].set_xlabel('Iteration')

        # Heat Price Plot
        sns.scatterplot(x=df.index, y='Heat Price (€)', data=df, ax=gs[0, 1], legend=False)
        gs[0, 1].set_xlabel('Iteration')

        # Electricity Price Plot
        sns.scatterplot(x=df.index, y='Electricity Price (€)', data=df, ax=gs[1, 0],
                        legend=False)
        gs[1, 0].set_xlabel('Iteration')

        # LCOH Plot
        sns.scatterplot(x='Iteration', y='LCOH (€/MWh)', data=last_df, ax=gs[1, 1], legend=False)
        gs[1, 1].set_xlabel('Realization')

        # NPV Plot
        sns.scatterplot(x='Iteration', y='NPV (€)', data=last_df, ax=gs[2, 0], legend=False)
        gs[2, 0].set_xlabel('Realization')

        
        # Replace all infinite values in the npv_df column with NaN
        # This ensures that seaborn/pandas will treat them as missing data and skip them during plotting,
        # avoiding the deprecated `use_inf_as_na` warning.
        npv_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # NPV Over Time Plot
        sns.lineplot(x='Time (yrs)', y='NPV (€)',
                     data=npv_df,
                     ax=gs[2, 1],
                     color='lightgray',
                     linewidth=1,
                     alpha=0.35,
                     estimator=None,
                     units='Iteration',
                     legend=False)

        # Compute NPV percentiles at each time step using agg() for each percentile
        npv_percentiles = npv_df.groupby('Time (yrs)')['NPV (€)'].agg(
            P10=lambda x: np.percentile(x, 10),
            P50=lambda x: np.percentile(x, 50),
            P90=lambda x: np.percentile(x, 90)
        ).reset_index()

        payback_p10 = self.calc_payback_period(df=npv_percentiles, column="P10", verbose=verbose, warning=warning)
        payback_p50 = self.calc_payback_period(df=npv_percentiles, column="P50", verbose=verbose, warning=warning)
        payback_p90 = self.calc_payback_period(df=npv_percentiles, column="P90", verbose=verbose, warning=warning)

        # Plot the percentile lines in the last subplot (gs[2, 1])
        ax = gs[2, 1]
        
        ax.plot(npv_percentiles['Time (yrs)'], npv_percentiles["P10"], label='P10', color='blue')
        ax.plot(npv_percentiles['Time (yrs)'], npv_percentiles["P50"], label='P50', color='red')
        ax.plot(npv_percentiles['Time (yrs)'], npv_percentiles["P90"], label='P90', color='green')
        
        if payback_p10 is not None:
            ax.axvline(x=payback_p10, color='blue', linestyle='--', linewidth=1.5, label=f"Payback_P10: {payback_p10:.1f} yr.")
        if payback_p50 is not None:
            ax.axvline(x=payback_p50, color='red', linestyle='--', linewidth=1.5, label=f"Payback_P50: {payback_p50:.1f} yr.")
        if payback_p90 is not None:
            ax.axvline(x=payback_p90, color='green', linestyle='--', linewidth=1.5, label=f"Payback_P90: {payback_p90:.1f} yr.")
            
        ax.set_xlabel('Time (yrs)')
        ax.set_ylabel('NPV (€)')
        ax.set_title('NPV Percentiles Over Time')
        ax.legend(loc='best',fontsize=8, ncol=2)

        plt.tight_layout()
        plt.savefig('MC_NPV_overview_plot.png')

        #--------------------Drawing NPV and LCOE Convergence plot (p10, p50, p90)--------------------
        fig, gs = plt.subplots(figsize=(9, 9), nrows=2, dpi=300, sharex=True)
        fig.suptitle('NPV & LCOE\nMC Convergence plot')

        xb = np.random.choice(last_df['NPV (€)'], (iterations, reps), replace=True)
        yb = 1 / np.arange(1, iterations + 1)[:, None] * np.cumsum(xb, axis=0)
        p10, p50, p90 = np.percentile(yb, [10, 50, 90], axis=1)
        gs[0].plot(np.arange(1, iterations + 1)[:, None], yb, c='grey', alpha=0.02)
        line1, = gs[0].plot(np.arange(1, iterations + 1), p10, c='blue', linewidth=1, label="p10")
        line2, = gs[0].plot(np.arange(1, iterations + 1), p50, c='red', linewidth=1, label="p50")
        line3, = gs[0].plot(np.arange(1, iterations + 1), p90, c='green', linewidth=1, label="p90")
        gs[0].set_ylabel('NPV (€)')

        xb = np.random.choice(last_df['LCOH (€/MWh)'], (iterations, reps), replace=True)
        yb = 1 / np.arange(1, iterations + 1)[:, None] * np.cumsum(xb, axis=0)
        p10, p50, p90 = np.percentile(yb, [10, 50, 90], axis=1)
        gs[1].plot(np.arange(1, iterations + 1)[:, None], yb, c='grey', alpha=0.02)
        gs[1].plot(np.arange(1, iterations + 1), p10, c='blue', linewidth=1, label="p10")
        gs[1].plot(np.arange(1, iterations + 1), p50, c='red', linewidth=1, label="p50")
        gs[1].plot(np.arange(1, iterations + 1), p90, c='green', linewidth=1, label="p90")
        gs[1].set_xlabel('Realization')
        gs[1].set_ylabel('LCOH (€/MWh)')

        # Create a single legend for the entire figure, with only p10, p50, and p90
        fig.legend([line1, line2, line3], ['p10', 'p50', 'p90'],
                   loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 0.94), fontsize=10)

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig('MC_NPV_Convergence_plot.png')
        
        print("\n****Monte Carlo simulation run succesfully!****\n")

        return mc_list

    def organize_input_data(self):
        """
        Organizes and converts initial data using well input parameters and conversion factors.

        Converts time, temperature, pressure, and flow rate columns based on coefficients in well_inputs.
        Exits if any expected columns are missing.

        Raises:
            Exception: If expected column names are missing from the input data.
        """
        try:
            self._obj["Time (yrs)"] = self.initial_data[self.well_inputs.time_column_name] * \
                                      self.well_inputs.CoEff_units[
                                          0]  # CoEff_units first item belongs to time conversion factor

            for i, well in enumerate(self.well_inputs.well_names):
                well_name = well + "_" + self.well_inputs.well_types[i]
                self._obj[well_name + "_T"] = self.initial_data[self.well_inputs.T_column_names[i]] + \
                                              self.well_inputs.CoEff_units[1]
                self._obj[well_name + "_P"] = self.initial_data[self.well_inputs.P_column_names[i]] * \
                                              self.well_inputs.CoEff_units[2]
                self._obj[well_name + "_Q"] = self.initial_data[self.well_inputs.Q_column_names[i]] * \
                                              self.well_inputs.CoEff_units[3]

        except Exception as e:
            print(
                f"Some of the column names are not included in the input file!\nDetails: {e}\nThe program is aborting...")
            sys.exit()

        if self.verbose:
            print("\n******* Organized Data File *******")
            print(self._obj.head())

    def check_exact_years(self):
        """
        Identifies and inserts missing exact year values in the 'Time (yrs)' column of the DataFrame. It also applies
        linear interpolation to fill in NaN values in the newly inserted rows based on existing data. Additionally,
        it creates a DataFrame containing only the rows that correspond to these newly inserted years.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to process. It must contain a column named 'Time (yrs)' with numeric values.
        
        Steps:
        1. Iterates through the 'Time (yrs)' column to find gaps in sequential year entries.
        2. For each missing year found, inserts a new row with the missing year and NaN values for other columns.
        3. Applies linear interpolation to fill NaN values across the DataFrame, using existing values to estimate the missing ones.
        4. Constructs a new DataFrame comprising only the rows with years that were inserted during the process.
        
        Returns:
        - df (pd.DataFrame): The modified original DataFrame with new rows inserted and NaN values interpolated.
        - interpolated_dates_df (pd.DataFrame): A DataFrame of only the newly inserted rows, where the interpolation has been applied.
        """

        year = 0
        rows_to_insert = []  # List to store new rows to be added
        inserted_years = []

        for ind, t in enumerate(self._obj['Time (yrs)']):
            if t == 0:
                continue

            while t > year + 1:
                year += 1
                if self.verbose:
                    print("Missing year:", year, "\tdf_index:", ind)

                # Create a new row with only 'Time (yrs)'
                new_row = {'Time (yrs)': year}
                rows_to_insert.append((ind, new_row))
                inserted_years.append(year)

            if t >= year:
                if t % 1 == 0:
                    year += 1

        # Add new rows
        for ind, row in reversed(rows_to_insert):  # Reverse to prevent index shifting
            self._obj = pd.concat([self._obj.iloc[:ind], pd.DataFrame([row]), self._obj.iloc[ind:]]).reset_index(
                drop=True)

        # ----------assign time column to index as day for doing weighted interpolation----------
        self._obj['Time_copy'] = self._obj['Time (yrs)']

        self._obj.index = pd.to_timedelta(self._obj['Time (yrs)'] * 365.25, unit='d')
        self._obj.index.name = 'Time (days)'

        self._obj['Time (yrs)'] = self._obj['Time_copy']
        self._obj.drop('Time_copy', axis=1, inplace=True)
        # ----------------------------------------------------------------------------------------

        # Apply weighted interpolation to Nan values in added rows
        self._obj.interpolate(method='index', inplace=True)

        # Create a DataFrame of modified rows
        interpolated_dates_df = self._obj[self._obj['Time (yrs)'].isin(inserted_years)]

        return interpolated_dates_df

    def assign_econ_periods(self):
        """
        Rounds up the 'Time (yrs)' column values to the nearest whole number and stores these values in a new
        column named 'econ_periods'. This function is useful for segmenting the data into economic periods based
        on years.

        Parameters:
        - df (pd.DataFrame): DataFrame with a numeric column named 'Time (yrs)'.

        Returns:
        - pd.DataFrame: The modified DataFrame with an added column 'econ_periods' containing the rounded-up year values.
        """
        # Check if 'Time (yrs)' column exists
        if 'Time (yrs)' not in self._obj.columns:
            raise ValueError("The DataFrame must contain a column named 'Time (yrs)'")

        # Round up the 'Time (yrs)' values and store in a new column
        self._obj['econ_periods'] = np.ceil(self._obj['Time (yrs)']).astype(int)

    def calc_per_discount_rate(self, mode=1):
        """Calculates the periodic discount rate and operational expense discount rate over time
       based on the specified mode.

       Parameters:
       ----------
       mode : int, optional
           Specifies the calculation mode for discount rates. Default is 1.

           - Mode 1: Uses the economic periods directly for the discount rate calculation.
             Assumes 'econ_periods' in the DataFrame represents the number of periods
             in years. This mode calculates the periodic discount rate and periodic
             operational expense discount rate based on full years.

           - Mode 2: Same as Excel xNPV function. Calculates the discount rates based on the exact elapsed time
             in days rather than whole economic periods. This mode converts 'Time (yrs)'
             from years to days, allowing a more precise calculation by dividing
             'Time (days)' by 365 to obtain yearly fractions."""

        r = self.econ_inputs.annual_discount_rate
        r_opex = self.econ_inputs.OpEx_annual_percent_CapEx

        if mode == 1:
            periodic_discount_rate = (1 + r) ** self._obj['econ_periods']
            self._obj['disc_rate'] = periodic_discount_rate

            periodic_opex_rate = (1 + r_opex) ** self._obj['econ_periods']
            self._obj['opex_disc_rate'] = periodic_opex_rate

        elif mode == 2:
            self._obj['Time (days)'] = (self._obj['Time (yrs)'] * 365.25).astype(float)
            periodic_discount_rate = (1 + r) ** (self._obj['Time (days)'] / 365)
            self._obj['disc_rate'] = periodic_discount_rate

            periodic_opex_rate = (1 + r_opex) ** (self._obj['Time (days)'] / 365)
            self._obj['opex_disc_rate'] = periodic_opex_rate

    def calculate_deltahours(self):
        # 'Time (yrs)' kolonundaki bir önceki yılla olan farkı hesapla
        temp_df = self._obj['Time (yrs)'] * 365.25 * pd.Timedelta('1 day')
        self._obj['Deltahours'] = temp_df.diff(1) / pd.Timedelta('1 hour')

    def overview_plot_input(self, interpolated_dates):
        T_col = self._obj.filter(like='_T').columns.tolist()
        P_col = self._obj.filter(like='_P').columns.tolist()
        Q_col = self._obj.filter(like='_Q').columns.tolist()
        dp_col = self._obj.filter(like='Pump dp (MPa)').columns.tolist()
        y_labels = ["Temperature (°C)", "Flowrate (m3/day)", "Pressure (bar-a)", "dp_pump (MPa)"]

        columns = [T_col, Q_col, P_col, dp_col]

        fig, ax = plt.subplots(2, 2, figsize=(12, 6), dpi=300, sharex=True)
        ax_list = fig.axes

        for i, col in enumerate(columns):
            for j, item in enumerate(col):
                self._obj.plot(x='Time (yrs)', y=item, ax=ax_list[i], legend=False)

                # if i < len(columns) - 1:  # dp_col does not exist ininterpolated_dates!
                #     ax_list[i].scatter(interpolated_dates['Time (yrs)'], interpolated_dates[item], marker="$\u25EF$",
                #                        color='red', zorder=5)

                ax_list[i].set_ylabel(y_labels[i],
                                      fontsize=12)
                ax_list[i].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)
                ax_list[i].tick_params(axis="both", which="both")

        fig_labels = self.well_inputs.well_names
        # fig_labels = np.insert(fig_labels, 1, 'interpolated data')
        # fig_labels.append('interpolated data')
        fig.legend(labels=fig_labels,
                   loc="lower center",
                   # bbox_to_anchor=(0.5, -0.04),
                   ncol=len(self.well_inputs.well_names) + 1,
                   fancybox=True,
                   shadow=False,
                   fontsize=9)

        plt.tight_layout()
        plt.savefig('Input_overview_plot.png')
        # plt.show()
        # plt.close(fig)

    def overview_plot_output(self):
        # fig, ax = plt.subplots(3, 2, figsize=(12, 8), dpi=300, sharex=True)
        # ax_list = fig.axes
        fig = plt.figure(figsize=(12, 8), dpi=300)
        gs = fig.add_gridspec(3, 2)
        ax_list = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

        ax_list[0].set_title('Pump Power (MW)')
        pump_power_col = self._obj.filter(like='Pump power (MW)').columns.tolist()
        for i, col in enumerate(pump_power_col):
            ax_list[0].plot(self._obj['Time (yrs)'], self._obj[col],
                            linestyle='-', label=self.well_inputs.well_names[i])
        ax_list[0].plot(self._obj['Time (yrs)'], self._obj['Pump power total (MW)'], linestyle='--', label='Total')
        ax_list[0].tick_params(axis="both", which="both")
        ax_list[0].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)
        ax_list[0].legend(loc='best', fontsize=7)

        ax_list[1].set_title('Produced Net Power (MW)')
        ax_list[1].plot(self._obj['Time (yrs)'], self._obj['Power net (MW)'])
        ax_list[1].tick_params(axis="both", which="both")
        ax_list[1].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)

        ax_list[2].set_title('Income (\u20ac)')
        ax_list[2].plot(self._obj['Time (yrs)'], self._obj['Income (\u20ac)'])
        ax_list[2].tick_params(axis="both", which="both")
        ax_list[2].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)

        ax_list[3].set_title('CF (M€)')
        ax_list[3].plot(self._obj['Time (yrs)'], self._obj['CF (€)'] * 10 ** -6)
        ax_list[3].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)

        ax_list[4].set_title('NPV (M€)')
        ax_list[4].plot(self._obj['Time (yrs)'], self._obj['NPV (€)'] * 10 ** -6)
        ax_list[4].scatter(self.payback_period_npv, 0, s=25, marker='o', facecolors='none', edgecolors='red',
                           linewidths=2)
        ax_list[4].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)
        ax_list[4].set_xlabel('Time (Years)')

        # Get last LCOH value
        last_time = self._obj['Time (yrs)'].iloc[-1]
        last_lcoh = self._obj['LCOH (€/MWh)'].iloc[-1]
        ax_list[5].set_title('LCOE (€/MWh)')
        ax_list[5].plot(self._obj['Time (yrs)'], self._obj['LCOH (€/MWh)'])
        ax_list[5].scatter(last_time, last_lcoh, s=25, marker='o', facecolors='none', edgecolors='red', linewidths=2)
        ax_list[5].grid(which='major', axis="both", linestyle='--', color='grey', alpha=0.3)
        ax_list[5].set_xlabel('Time (Years)')
        ax_list[5].set_ylim(0, self._obj['LCOH (€/MWh)'].iloc[-1] * 5)

        
        if self.payback_period_npv!=None:
            ax_list[4].annotate(f'Payback Period:{self.payback_period_npv:.1f} years',
                                xy=(self.payback_period_npv, 0),  # Son noktanın koordinatları
                                xytext=(45, -45),  # Offset: x ve y'de 15 birim kaydırma
                                textcoords='offset points',  # Ofsetin hangi birimle yapılacağı
                                arrowprops=dict(arrowstyle="->", color='black'),  # İsteğe bağlı: bir ok ekleyebilirsiniz
                                ha='center', va='bottom',  # Horizontally and vertically align the text
                                bbox=dict(boxstyle="round,pad=0.5", fc="green", alpha=0.5),
                                # Boxstyle özelliklerini uygulama
                                fontsize=10)

        ax_list[5].annotate(f'{last_lcoh:.2f} €/MWh\n@ {last_time:.1f} year',
                            xy=(last_time, last_lcoh),  # Son noktanın koordinatları
                            xytext=(15, 15),  # Offset: x ve y'de 15 birim kaydırma
                            textcoords='offset points',  # Ofsetin hangi birimle yapılacağı
                            arrowprops=dict(arrowstyle="->", color='black'),  # İsteğe bağlı: bir ok ekleyebilirsiniz
                            ha='center', va='bottom',  # Horizontally and vertically align the text
                            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
                            # Boxstyle özelliklerini uygulama
                            fontsize=10)

        plt.tight_layout()
        plt.savefig('Output_overview_plot.png')

        # plt.show()
        # plt.close(fig)
        
        
    def add_cost(self, cost_item: str, unit_cost: float, repl_freq: float, quantity: int = 1):
        """
        Add or update a cost component in the cost_comp dictionary.
    
        :param cost_item: Name of the cost component.
        :param unit_cost: Unit cost of the component (in euros).
        :param repl_freq: Replacement frequency of the component (in years).
        :param quantity: Number of units (default is 1).
        """
        self.econ_inputs.cost_comp[cost_item] = {
            "unit_cost": unit_cost,
            "quantity": quantity,
            "replacement_freq": repl_freq}
        
        print(f"Cost component '{cost_item}' added successfully.")
    
    def remove_cost(self, cost_item: str):
        """
        Remove a cost component from the cost_comp dictionary.
    
        :param cost_item: Name of the cost component to remove.
        """
        if cost_item in self.econ_inputs.cost_comp:
            del self.econ_inputs.cost_comp[cost_item]
            print(f"Cost component '{cost_item}' removed successfully.")
        else:
            print(f"Cost component '{cost_item}' not found.")


#%%
# ------------------------------------------------Ancillary Functions------------------------------------------------
def calculate_well_costs(well_costs=[500000], drill_depth_list=[500],
                         learning_curve=False, drill_depth_calc=False, verbose=False, **kwargs):
    if verbose:
        print('\n\n********Drilling field function started********\n')

    if drill_depth_calc:
        if learning_curve:
            use_learning_curve = None
            lc_max, lc_med, lc_min = learning_curves(x=np.arange(0, len(drill_depth_list), 1), verbose=verbose)
            if learning_curve == 'max':
                use_learning_curve = lc_max
            if learning_curve == 'med':
                use_learning_curve = lc_med
            if learning_curve == 'min':
                use_learning_curve = lc_min
            if isinstance(use_learning_curve, np.ndarray):
                if verbose:
                    print('\tCost for drilling field with: \n\t\t%s wells'
                          '                                 \n\t\tto depths %s with '
                          '                                  \n\t\t%s learning curve' % (
                              len(drill_depth_list), drill_depth_list, str.upper(learning_curve)))
                wellcosts = sum(
                    [drillingcostnl(depth) * use_learning_curve[i] for i, depth in enumerate(drill_depth_list)])

        else:  # if learning curve false
            if verbose:
                print('\tCost for drilling field with: \n\t\t%s wells'
                      '                                 \n\t\tto depths %s with '
                      '                                  \n\t\tNO learning curve' % (
                          len(drill_depth_list), drill_depth_list))
            wellcosts = sum([drillingcostnl(depth) for depth in drill_depth_list])

    else:  # if well costs are listed by the user and drill_depth_calc assigned False
        wellcosts = sum(well_costs)

    if verbose:
        print('\tField drilling costs: %s' % wellcosts)

    return wellcosts


def drillingcostnl(depth):
    """
    Calculate the cost of drilling as a function of depth
    Reference source:
        https://www.thermogis.nl/en/economic-model

    :param depth: float
        measured depth along hole in meters

    :return: float
        costs in euros
    """
    drilling_cost_nl = 375000 + 1150 * depth + 0.3 * depth ** 2
    return (drilling_cost_nl)


def learning_curves(x=np.arange(0, 50, 1), ts=[1, 3.5, 3.5], yends=[0.7, 0.75, 0.9], verbose=False, **kwargs):
    lcurves = []
    for t, yend in zip(ts, yends):
        # lcurves.append(learning_curve(x,t,1,yend))
        lcurves.append(learning_curve(x, t, yend))

    return (lcurves)


# def learning_curve(wn, t, y0, yend):
def learning_curve(wn, t, yend):
    """

    :param wn: number of wells
    :param t: decay constant
    :param y0: intial y value
    :param yend: final y value
    :return: exponential decay function
    """

    # learning_curve = y0 + (yend*y0 - y0) * (1 - np.exp(-wn / t))
    learning_curve = 1 + (yend - 1) * (1 - np.exp(-wn / t))
    return (learning_curve)


def units(time: int, T: int, P: int, Q: int) -> list:
    """
    Select units and their conversion factors based on numerical inputs for Time, Temperature (T), Pressure (P), and Flow rate (Q).

    Args:
    time (int): A number for the unit of time
                (1: Year, 2: Month, 3: Day, 4: Hour, 5: Minute, 6: Second).
    T (int): A number for the unit of temperature
             (1: Celsius, 2: Kelvin).
    P (int): A number for the unit of pressure
             (1: Bar, 2: MPa, 3: Atm, 4: kPa, 5: Pa, 6: PSI).
    Q (int): A number for the unit of flow rate
             (1: m3/day, 2: m3/h, 3: m3/min, 4: m3/s, 5: cm3/day, 6: cm3/h,
              7: cm3/min, 8: cm3/s, 9: l/day, 10: l/h, 11: l/min, 12: l/s).

    Returns:
    list: A list of conversion factors based on selected units in the order
          [time_factor, temperature_factor, pressure_factor, flow_rate_factor].

    Notes:
    Default units: Year, Celsius, Bar, m3/day.
    All conversions are done based on default units.
    """

    # Dictionary of all units with conversion factors
    units_dict = {
        "Time": {
            "Year": 1,
            "Month": (1 / 12),
            "Day": (1 / 365.25),
            "Hour": (1 / (365.25 * 24)),
            "Minute": (1 / (365.25 * 24 * 60)),
            "Second": (1 / (365.25 * 24 * 60 * 60))
        },
        "Temperature": {
            "Celsius": 0,
            "Kelvin": -273.15
        },
        "Pressure": {
            "Bar": 1,
            "MPa": 10,
            "Atm": 1.01325,
            "kPa": 0.01,
            "Pa": 1e-5,
            "PSI": 0.9576052
        },
        "Flow rate": {
            "m3/day": 1,
            "m3/h": 24,
            "m3/min": 1440,
            "m3/s": 86400,
            "cm3/day": 1e-6,
            "cm3/h": 2.4e-5,
            "cm3/min": 0.00144,
            "cm3/s": 0.0864,
            "l/day": 0.001,
            "l/h": 0.024,
            "l/min": 1.44,
            "l/s": 86.4
        }
    }

    # Selecting unit conversions based on inputs
    unit_conversions = [
        units_dict["Time"].get(time),
        units_dict["Temperature"].get(T),
        units_dict["Pressure"].get(P),
        units_dict["Flow rate"].get(Q)
    ]

    return unit_conversions


def calculate_heatpower_columns(row, well):
    """
    This function calculates Heat Power (MW) for each row of the well data.
    """
    # get T, P, Q from df
    # Q should be positive for injection and negative for production
    T_kelvin = row[well + "_T"] + 273.15  # Celsius to Kelvin
    P = row[well + "_P"] * 0.1  # bar to MPa, Default P:bar
    Q = row[well + "_Q"]  # Default Q:m3/day

    # Calculating density and enthalpy with IAPWS97
    water_properties = IAPWS97(T=T_kelvin, P=P)
    density = water_properties.rho  # kg/m^3
    enthalpy = water_properties.h  # kJ/kg

    # Convert Q from m3/day to m3/s by dividing by 86400
    Q = Q / 86400  # Now Q is in m3/s

    # Calculate power in kW then convert to MW
    power = Q * density * enthalpy  # kJ/s or kW
    power = power / 1000  # Convert kW to MW

    # print(well + "_T :%5.2f"%(row[well + "_T"]),'\nEnthalpy (kj/kg):%5.2f'%enthalpy,\
    #       '\nDensity (kg/m3):%5.2f'%density,'\nFlowrate (m3/day):%5.2f'%Q,'\nPower (MW):%5.2f'%power)
    return power


def read_files(use_browser: bool = False, data_file_path=None, input_file_path=None):
    """
      Reads data and input files based on user selection or provided file paths.

      Args:
          use_browser (bool): If True, file dialogs will be opened to select the files.
          data_file_path (str): Path to the data file if provided directly.
          input_file_path (str): Path to the input file if provided directly.

      Returns:
          econ (pd.DataFrame): DataFrame containing the processed data if successful.

      Notes:
          The function expects the input file to contain sheets named 'Units', 'Well_Inputs', 'Cost_Inputs', and 'Econ_Inputs'.
          In case of errors while reading files, the function will print an error message and exit the program.
      """
    if use_browser:
        data_file_path = open_file_dialog(dialog_type="open",
                                          title="Please select the data file to open")
        input_file_path = open_file_dialog(dialog_type="open",
                                           title="Please select the parameter file to open")
    else:
        data_file_path = data_file_path
        input_file_path = input_file_path

    # ------------Read data & inputs------------
    try:
        data = pd.read_excel(data_file_path)
    except Exception as e:
        print(
            f"Error: The program could not read Time Series input file!\nControl the file path ({data_file_path}) or the file!\nDetails: {e}")
        sys.exit()

    # Read well inputs and economic inputs from separate sheets
    try:
        units = pd.read_excel(input_file_path, sheet_name='Units')
        well_inputs = pd.read_excel(input_file_path, sheet_name='Well_Inputs')
        cost_inputs = pd.read_excel(input_file_path, sheet_name='Cost_Inputs')
        econ_inputs = pd.read_excel(input_file_path, sheet_name='Econ_Inputs')
    except Exception as e:
        print(
            f"Error: The program could not read input file!\nControl the file path ({input_file_path}) or the file!\nDetails: {e}")
        sys.exit()

    # ------------Create an econ object and assign parameters------------
    econ = pd.DataFrame().gte

    econ.initial_data = data
    econ.assign_units(units)
    econ.assign_well_inputs(well_inputs)
    econ.assign_cost_inputs(cost_inputs)
    econ.assign_econ_inputs(econ_inputs)

    return econ


def open_file_dialog(dialog_type="open", title="Select a file"):
    import tkinter as tk
    from tkinter import filedialog

    # Create main Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide main window

    # Define the file types
    filetypes = [("Excel files", "*.xlsx"), ("Old Excel files", "*.xls"), ("CSV files", "*.csv")]

    # Get the directory of the current script
    initial_dir = os.path.dirname(os.path.abspath(__file__))

    # select file path
    if dialog_type == "open":
        file_path = filedialog.askopenfilename(title=title, initialdir=initial_dir, filetypes=filetypes)
    elif dialog_type == "save":
        file_path = filedialog.asksaveasfilename(title=title, initialdir=initial_dir, defaultextension=".xlsx",
                                                 filetypes=filetypes)
    return file_path


def pump_cost(pump_power_MW, pump_pres_bar,
              K1=3.3892, K2=0.0536, K3=0.1538,
              C1=-0.3935, C2=0.3957, C3=-0.0023,
              Fm=1.6, B1=1.8900, B2=1.3500):
    pump_power_kW = pump_power_MW * 1e3
    initial_cost = 10 ** (K1 + K2 * np.log10(pump_power_kW) + K3 * np.log10(pump_power_kW) ** 2)
    Fp = 10 ** (C1 + C2 * np.log10(pump_pres_bar) + C3 * np.log10(pump_pres_bar) ** 2)

    pump_cost_ = 2 * initial_cost * (B1 + (B2 * Fm * Fp))
    return pump_cost_
