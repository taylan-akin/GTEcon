from GTEcon_module import GTEcon, units
from utils.hx import HeatExchangerCostModel

'''
Demonstrates how to use the GTEcon module for techno-economic analysis of HT-ATES systems.
'''

#%% Example 1: Automatic loading of all inputs from Excel files

# Instantiate the GTEcon model with well-operation data and parameter file
econ = GTEcon(well_operation_path="Input Example/Well operation data.xlsx",
              params_path="Input Example/Parameter Inputs.xlsx")

# ------------Run GTEcon------------
econ.run(plotting=True)
print(f"LCOH: {econ.lcoe:.2f},\nNPV: {econ.npv:.2f}")

# ------------Optional Monte Carlo simulation------------
mc_list = econ.mc_econ(params={"Discount Rate": (0.1, 0.1 * 0.1),
                                "Heat Price": (50, 50 * 0.1),
                                "Electricity Price": (100, 100 * 0.1)},
                       iterations=1000,
                       warning=False,
                       verbose=False)

#%% Example 2: Manual assignment of inputs

# Instantiate GTEcon model with well-operation data only;
# no params_path provided → user must assign units, wells, costs & econ inputs manually
econ = GTEcon(well_operation_path="Input Example/Well operation data.xlsx"
              # params_path omitted -> user must assign all inputs manually
              )

# Assign well operation data automatically loaded in __init__
# Now define well and unit parameters
econ.well_inputs.well_names=['H1', 'L1']
econ.well_inputs.well_types=["H", "W"]  # Hot Well:H   Warm Well:W
econ.well_inputs.well_depths_m=[180, 180]
econ.well_inputs.reservoir_depth_tvd_m=[150, 150]
econ.well_inputs.pump_efficiency=[0.5, 0.5]
econ.well_inputs.T_column_names=['H1 : temperature (K)', 'L1 : temperature (K)']
econ.well_inputs.P_column_names=['H1 : BHP (bar)', 'L1 : BHP (bar)']
econ.well_inputs.Q_column_names=['H1 : water rate (m3/day)', 'L1 : water rate (m3/day)']
econ.well_inputs.time_column_name='time'
econ.well_inputs.CoEff_units=units("Day", "Kelvin", "Bar", "m3/day")
econ.well_inputs.pipeline_pressure=3    # bar-a, pressure of surface pipeline at the wellhead

# Economic input parameters
econ.annual_discount_rate=0.1
econ.heat_price_euro_MWh=50
econ.electricity_price_euro_MWh=100

# Define cost components
econ.add_cost(cost_item="Wells", unit_cost=250000,quantity=2,repl_freq=0)
econ.add_cost(cost_item="Well Pumps", unit_cost=76129,quantity=4,repl_freq=0)
econ.add_cost(cost_item="Controls and Electrics", unit_cost=241034,quantity=1,repl_freq=0)
econ.add_cost(cost_item="Heat Pump", unit_cost=3258696,quantity=1,repl_freq=0)
econ.add_cost(cost_item="Maintenance and Management of Pumps", unit_cost=13276.7,quantity=4,repl_freq=1)
econ.add_cost(cost_item="Water Treatment", unit_cost=13921,quantity=1,repl_freq=1)
econ.add_cost(cost_item="Leasing Fee", unit_cost=4464,quantity=1,repl_freq=1)
econ.add_cost(cost_item="Heat Exchanger",
              unit_cost=HeatExchangerCostModel("flat_plate").compute_cost(10),
              quantity=1,
              repl_freq=0)


# Run model and display results
econ.run(plotting=True)
print(f"LCOH: {econ.lcoe:.2f},\nNPV: {econ.npv:.2f}")

# Optional Monte Carlo simulation
mc_list = econ.mc_econ(params={"Discount Rate": (0.1, 0.1 * 0.1),
                                "Heat Price": (50, 50 * 0.1),
                                "Electricity Price": (100, 100 * 0.1)},
                       iterations=1000)