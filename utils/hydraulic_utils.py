import math
import pandas as pd
from scipy.optimize import fsolve
from CoolProp.CoolProp import PropsSI

def colebrook_scipy(f, Re, D, epsilon):
    """
    Colebrook-White equation function for the friction factor.
    
    This function returns the residual of the Colebrook-White equation:
      1/sqrt(f) + 2 * log10((epsilon/(3.7*D)) + (2.51/(Re*sqrt(f)))) = 0
    
    Parameters:
      f       : Friction factor (scalar)
      Re      : Reynolds number
      D       : Pipe inner diameter (m)
      epsilon : Pipe roughness (m)
      
    Returns:
      The residual of the Colebrook-White equation.
    """
    # Convert f to a scalar if it's an array-like value.
    f_scalar = f[0] if hasattr(f, '__iter__') else f
    return 1.0 / math.sqrt(f_scalar) + 2.0 * math.log10((epsilon / (3.7 * D)) + (2.51 / (Re * math.sqrt(f_scalar))))

def solve_colebrook(Re, D, epsilon, initial_guess=0.02):
    """
    Solves the Colebrook-White equation for the friction factor using SciPy's fsolve.
    
    Parameters:
      Re           : Reynolds number
      D            : Pipe inner diameter (m)
      epsilon      : Pipe roughness (m)
      initial_guess: Initial guess for the friction factor (default is 0.02)
      
    Returns:
      f_solution   : Solved friction factor.
    """
    f_solution, = fsolve(colebrook_scipy, initial_guess, args=(Re, D, epsilon))
    return f_solution

def hydraulic_loss(flw, T, D, L, epsilon, P=1.01325):
    """
    Calculates the hydraulic pressure loss for a borehole heat exchanger (BHE)
    for each corresponding pair of volumetric flow rate (flw, in m³/day) and 
    fluid temperature (T, in °C). The function converts the volumetric flow rate
    to a mass flow rate (kg/s) based on the water density computed using CoolProp,
    given the temperature and pressure. The pressure (P) is provided in bar 
    (default is 1.01325 bar, atmospheric pressure) and is converted to Pa internally.
    
    The hydraulic loss (ΔP) is calculated using the Darcy-Weisbach equation:
      ΔP = f * (8 * m_dot² * L) / (π² * D⁵ * ρ)
      
    where:
      Re = (4 * m_dot) / (π * D * μ)
      
    and the friction factor f is obtained by solving the Colebrook-White equation:
      1/sqrt(f) + 2 * log10((epsilon/(3.7*D)) + (2.51/(Re*sqrt(f)))) = 0
    
    Parameters:
      flw     : Volumetric flow rates in m³/day (list, array, or Pandas Series)
      T       : Fluid temperatures in °C (list, array, or Pandas Series)
      D       : Pipe inner diameter (m)
      L       : Pipe length (m) (e.g., 2 times the borehole depth)
      epsilon : Pipe roughness (m)
      P       : Fluid pressure in bar (default: 1.01325 bar)
      
    Returns:
      result_df : A pandas DataFrame with the computed values for each input:
                  'delta_P' : Hydraulic pressure loss in bar,
                  'Re'      : Reynolds number,
                  'f'       : Friction factor,
                  'rho'     : Fluid density (kg/m³),
                  'mu'      : Fluid dynamic viscosity (Pa·s).
    """
    # Prepare lists to store computed results for each input pair
    delta_P_list = []
    Re_list = []
    f_list = []
    rho_list = []
    mu_list = []
    
    # Convert pressure from bar to Pa (1 bar = 1e5 Pa)
    P_Pa = P * 1e5
    
    # Iterate over each pair of flow and temperature
    for flow_m3_day, T_celsius in zip(flw, T):
        # Convert temperature to Kelvin
        T_K = T_celsius + 273.15
        
        # Calculate water properties using CoolProp
        rho = PropsSI('D', 'T', T_K, 'P', P_Pa, 'Water')
        mu = PropsSI('VISCOSITY', 'T', T_K, 'P', P_Pa, 'Water')
        
        if flow_m3_day==0:           #If there is no flow, friction loss will be zero
            delta_P_list.append(0)
            Re_list.append(None)
            f_list.append(None)
        else:
            # Convert volumetric flow rate from m³/day to m³/s
            flow_m3_s = abs(flow_m3_day) / 86400.0
            
            # Convert volumetric flow rate to mass flow rate (kg/s)
            m_dot = flow_m3_s * rho
            
            # Calculate Reynolds number
            Re_value = 4 * m_dot / (math.pi * D * mu)
            
            # Solve for friction factor using the Colebrook-White equation
            friction_factor = solve_colebrook(Re_value, D, epsilon)
            
            # Calculate hydraulic pressure loss (in Pa) using the Darcy-Weisbach equation
            delta_P_Pa = friction_factor * (8 * m_dot**2 * L) / (math.pi**2 * D**5 * rho)
            
            # Convert pressure loss from Pa to bar
            delta_P_bar = delta_P_Pa / 1e5
            
            # Append computed values to the lists
            delta_P_list.append(delta_P_bar)
            Re_list.append(Re_value)
            f_list.append(friction_factor)
            
            
        rho_list.append(rho)
        mu_list.append(mu)
    
    # Create a DataFrame with the computed results
    result_df = pd.DataFrame({
        'delta_P': delta_P_list,
        'Re': Re_list,
        'f': f_list,
        'rho': rho_list,
        'mu': mu_list
    })
    
    return result_df


def hydraulic_loss_2(flow_m3_day, rho, D, mu, epsilon, L):
    # Convert volumetric flow rate from m³/day to m³/s
    flow_m3_s = abs(flow_m3_day) / 86400.0
    
    # Convert volumetric flow rate to mass flow rate (kg/s)
    m_dot = flow_m3_s * rho
    
    # Calculate Reynolds number
    Re_value = 4 * m_dot / (math.pi * D * mu)
    
    # Solve for friction factor using the Colebrook-White equation
    friction_factor = solve_colebrook(Re_value, D, epsilon)
    
    # Calculate hydraulic pressure loss (in Pa) using the Darcy-Weisbach equation
    delta_P_Pa = friction_factor * (8 * m_dot**2 * L) / (math.pi**2 * D**5 * rho)
    
    # Convert pressure loss from Pa to bar
    delta_P_bar = delta_P_Pa / 1e5
    
    return delta_P_bar

# Example usage:
# if __name__ == "__main__":
#     # Example flow rates in m³/day and temperatures in °C as separate lists
#     flw = [100, 150, 200]  # volumetric flows in m³/day
#     T = [50, 55, 60]       # corresponding temperatures in °C
    
#     data = {'flow': [1254, 1500, 2000],  # example volumetric flows in m³/day
#             'T': [80, 85, 90]}         # corresponding temperatures in °C
#     df_input = pd.DataFrame(data)
    
#     # Define constant parameters:
#     D = 0.15          # m, pipe inner diameter
#     borehole_depth = 68  # m
#     L = 2 * borehole_depth  # m, total pipe length (e.g., for a U-tube configuration)
#     epsilon = 3e-6     # m, pipe roughness
    
#     # Compute hydraulic loss results (default pressure of 1.01325 bar is used)
#     # results = hydraulic_loss(flw, T, D, L, epsilon)
#     results = hydraulic_loss(df_input["flow"], df_input["T"], D, L, epsilon)
    
#     # Print the results DataFrame
#     print(results)
