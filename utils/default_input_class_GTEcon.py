class well_inputs():
    """
   A class to handle well inputs for techno-economic assessments.

   :param well_names: List of well names.
   :param well_types: List of well types ("H" for Hot Well, "W" for Warm Well).
   :param well_depths_m: List of well depths (in meters).
   :param reservoir_depth_tvd_m: List of reservoir depths (True Vertical Depth in meters).
   :param pump_efficiency: List of pump efficiencies (as a decimal, e.g., 0.5 for 50%).
   :param time_column_name: Name of the time column.
   :param T_column_names: List of temperature column names.
   :param P_column_names: List of bottom hole pressure (BHP) column names.
   :param Q_column_names: List of flow rate column names.
   :param CoEff_units: List of coefficient units (e.g., ["Year", "Celsius", "Bar", "m³/day"]).
   :param pipeline_pressure: Pressure of surface pipeline at the wellhead (in bar absolute).
   """
    def __init__(self,
                 well_names=['H_1', 'W_1'],
                 well_types=["H", "W"],  # Hot Well:H   Warm Well:W
                 well_depths_m=[2500, 2500],
                 reservoir_depth_tvd_m=[2000, 2050],
                 pump_efficiency=[0.5, 0.5],
                 time_column_name='time',
                 T_column_names=['I1 : temperature (K)', 'P1 : temperature (K)'],
                 P_column_names=['I1 : BHP (bar)', 'P1 : BHP (bar)'],
                 Q_column_names=['I1 : water rate (m3/day)', 'P1 : water rate (m3/day)'],
                 CoEff_units=["Year", "Celsius", "Bar", "m3/day"],
                 pipeline_pressure=0  # bar-a, pressure of surface pipeline at the wellhead
                 ):
        self.well_names = well_names
        self.well_types = well_types
        self.well_depths_m = well_depths_m
        self.reservoir_depth_tvd_m = reservoir_depth_tvd_m
        self.pump_efficiency = pump_efficiency
        self.time_column_name = time_column_name
        self.T_column_names = T_column_names
        self.P_column_names = P_column_names
        self.Q_column_names = Q_column_names
        self.CoEff_units = CoEff_units
        self.pipeline_pressure = pipeline_pressure


class econ_inputs():
    """
    A class to handle economic inputs for techno-economic assessments.

    :param cost_comp: Dictionary containing component costs and details.
                      Keys are component names, values are dictionaries with:
                      - unit_cost: Cost per unit of the component (in euros).
                      - quantity: Number of units.
                      - replacement_freq: Replacement frequency (in years).
    :param OpEx_annual_percent_CapEx: Annual operational expenditure as a percentage of capital expenditure (e.g., 0.05 for 5%).
    :param annual_discount_rate: Annual discount rate (as a decimal, e.g., 0.1 for 10%).
    :param heat_price_euro_MWh: Price of heat (in euros per MWh).
    :param electricity_price_euro_MWh: Price of electricity (in euros per MWh).
    :param AbEx_per_well: Abandonment expenditure per well (in euros).
    :param surface_piping_cost_euro_m: Cost of surface piping per meter (in euros).
    :param unique_well_cost: Boolean flag indicating if well costs are unique.
    :param calculate_well_cost: Boolean flag indicating if well costs should be calculated.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 cost_comp={},
                 OpEx_annual_percent_CapEx=0,
                 annual_discount_rate=0.1,
                 heat_price_euro_MWh=50,
                 electricity_price_euro_MWh=100,
                 AbEx_per_well=0,
                 surface_piping_cost_euro_m=0,
                 unique_well_cost=False,
                 calculate_well_cost=False,
                 **kwargs):
        self.cost_comp = cost_comp
        self.OpEx_annual_percent_CapEx = OpEx_annual_percent_CapEx
        self.annual_discount_rate = annual_discount_rate
        self.heat_price_euro_MWh = heat_price_euro_MWh
        self.electricity_price_euro_MWh = electricity_price_euro_MWh
        self.AbEx_per_well = AbEx_per_well
        self.surface_piping_cost_euro_m = surface_piping_cost_euro_m
        self.unique_well_cost = unique_well_cost
        self.calculate_well_cost = calculate_well_cost
