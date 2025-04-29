import math

class HeatExchangerCostModel:
    """
    Heat Exchanger Cost Model.
    
    Parameters:
      exchanger_type (str): 'shell_and_tube' or 'flat_plate'
      cost_method (str): The calculation method to use. Default is "default" (general form). Alternatively,
                         a valid Table A2 label (e.g., "a2a", "a2b", etc.) can be provided.
    
    This class calculates the cost of a heat exchanger using two different methods:
    
      1. "default": The general form:
         C = log(A) + a * A² + b * A + c
         - For Shell-and-Tube exchangers:
               a = -0.06395, b = 947.2, c = 227.9
         - For Flat Plate exchangers:
               a = 0.2581,  b = 891.7, c = 26050  (i.e., 2.605×10⁴)
      
      2. Table A2 equations: Various correlations provided in Table A2 of the paper,
         which can be selected by specifying a valid equation label (e.g., "a2a", "a2b", …).
         For Flat Plate exchangers, additional equations ("a2w", "a2x", "a2y") are available.
      
    Reference (main paper):
      Shamoushaki, M., Niknam, P. H., Talluri, L., Manfrida, G., & Fiaschi, D. (2021). 
      Development of Cost Correlations for the Economic Assessment of Power Plant Equipment. 
      Energies, 14, 2665. https://doi.org/10.3390/en14092665
      
    Table A2 Equations and Corresponding References (APA format):
      
      - A2a: 
        Zare, V. (2015). A comparative exergoeconomic analysis of different ORC configurations 
        for binary geothermal power plants. Energy Conversion and Management, 105, 127–138. 
        https://doi.org/10.1016/j.enconman.2015.xxxx
      
      - A2b: 
        Sadeghi, M., Chitsaz, A., Mahmoudi, S., & Rosen, M. A. (2015). Thermoeconomic optimization 
        using an evolutionary algorithm of a trigeneration system driven by a solid oxide fuel cell. 
        Energy, 89, 191–204. https://doi.org/10.1016/j.energy.2015.xxxx
      
      - A2c: 
        (Same as A2b)
      
      - A2d: 
        Sadeghi, M., Chitsaz, A., Mahmoudi, S., & Rosen, M. A. (2015); 
        Khani, L., Mahmoudi, S. M. S., Chitsaz, A., & Rosen, M. A. (2016). 
        [Combined references]
      
      - A2e: 
        Aminyavari, M., Najafi, B., Shirazi, A., & Rinaldi, F. (2014). Exergetic, economic and 
        environmental (3E) analyses, and multi-objective optimization of a CO₂/NH₃ cascade refrigeration 
        system. Applied Thermal Engineering, 65, 42–50. https://doi.org/10.1016/j.applthermaleng.2014.xxxx
      
      - A2f: 
        Marandi, S., Mohammadkhani, F., & Yari, M. (2019). An efficient auxiliary power generation system 
        for exploiting hydrogen boil-off gas (BOG) cold exergy based on PEM fuel cell and two-stage ORC: 
        Thermodynamic and exergoeconomic viewpoints. Energy Conversion and Management, 195, 502–518. 
        https://doi.org/10.1016/j.enconman.2019.xxxx
      
      - A2g: 
        Wang, X., & Dai, Y. (2016). Exergoeconomic analysis of utilizing the transcritical CO₂ cycle and 
        the ORC for a recompression supercritical CO₂ cycle waste heat recovery: A comparative study. 
        Applied Energy, 170, 193–207. https://doi.org/10.1016/j.applenergy.2016.xxxx
      
      - A2h, A2i, A2j: 
        Mohammadi, A., Ashouri, M., Ahmadi, M. H., Bidi, M., Sadeghzadeh, M., & Ming, T. (2018). 
        Thermoeconomic analysis and multiobjective optimization of a combined gas turbine, steam, and 
        organic Rankine cycle. Energy Science & Engineering, 6, 506–522. 
        https://doi.org/10.1002/xxxxxxxx
      
      - A2k: 
        Akrami, E., Chitsaz, A., Nami, H., & Mahmoudi, S. (2017). Energetic and exergoeconomic assessment 
        of a multi-generation energy system based on indirect use of geothermal energy. Energy, 124, 
        625–639. https://doi.org/10.1016/j.energy.2017.xxxx
      
      - A2l: 
        Khosravi, H., Salehi, G. R., & Azad, M. T. (2019). Design of structure and optimization of organic 
        Rankine cycle for heat recovery from gas turbine: The use of 4E, advanced exergy and advanced 
        exergoeconomic analysis. Applied Thermal Engineering, 147, 272–290. https://doi.org/10.1016/j.applthermaleng.2019.xxxx
      
      - A2m: 
        Lecompte, S., Huisseune, H., Van den Broek, M., De Schampheleire, S., & De Paepe, M. (2013). 
        Part load based thermo-economic optimization of the Organic Rankine Cycle (ORC) applied to a 
        combined heat and power (CHP) system. Applied Energy, 111, 871–881. https://doi.org/10.1016/j.applenergy.2013.xxxx
      
      - A2n: 
        Zoghi, M., Habibi, H., Chitsaz, A., Javaherdeh, K., & Ayazpour, M. (2019). Exergoeconomic analysis 
        of a novel trigeneration system based on organic quadrilateral cycle integrated with cascade 
        absorption-compression system for waste heat recovery. Energy Conversion and Management, 198, 
        111818. https://doi.org/10.1016/j.enconman.2019.xxxx
      
      - A2o: 
        Mohammadkhani, F., & Yari, M. (2019). A 0D model for diesel engine simulation and employing a 
        transcritical dual loop Organic Rankine Cycle (ORC) for waste heat recovery from its exhaust and 
        coolant: Thermodynamic and economic analysis. Applied Thermal Engineering, 150, 329–347. 
        https://doi.org/10.1016/j.applthermaleng.2019.xxxx
      
      - A2p: 
        Gholizadeh, T., Vajdi, M., & Mohammadkhani, F. (2019). Thermodynamic and thermoeconomic analysis 
        of basic and modified power generation systems fueled by biogas. Energy Conversion and Management, 
        181, 463–475. https://doi.org/10.1016/j.enconman.2019.xxxx
      
      - A2q: 
        Smith, R. (2005). Chemical Process: Design and Integration. Hoboken, NJ, USA: John Wiley & Sons.
      
      - A2r: 
        Mosaffa, A., Farshi, L. G., Ferreira, C. I., & Rosen, M. (2016). Exergoeconomic and environmental 
        analyses of CO₂/NH₃ cascade refrigeration systems equipped with different types of flash tank intercoolers. 
        Energy Conversion and Management, 117, [page numbers]. https://doi.org/10.1016/j.enconman.2016.xxxx
      
      - A2s: 
        Reyhani, H. A., Meratizaman, M., Ebrahimi, A., Pourali, O., & Amidpour, M. (2016). Thermodynamic and 
        economic optimization of SOFC-GT and its cogeneration opportunities using generated syngas from heavy fuel oil gasification. 
        Energy, 107, 141–164. https://doi.org/10.1016/j.energy.2016.xxxx
      
      - A2t, A2u, A2x, A2y (Flat Plate): 
        Khosravi, H., Salehi, G. R., & Azad, M. T. (2019). Design of structure and optimization of organic Rankine cycle for heat recovery from gas turbine: 
        The use of 4E, advanced exergy and advanced exergoeconomic analysis. Applied Thermal Engineering, 147, 272–290. 
        https://doi.org/10.1016/j.applthermaleng.2019.xxxx
      
      - A2v, A2w (Flat Plate): 
        Vatavuk, W. M. (1995). A potpourri of equipment prices-Part 1. Chemical Engineering, 102, 68. 
        [DOI if available]
      
    Note: For some references, the DOI or complete details may not be provided in the source material.
    
    """
    
    def __init__(self, exchanger_type: str, cost_method: str = "default"):
        self.exchanger_type = exchanger_type.lower()
        self.cost_method = cost_method.lower()
        
        # Assign coefficients for the general form ("default" method)
        if self.exchanger_type in ['shell_and_tube', 'shell and tube']:
            self.a = -0.06395
            self.b = 947.2
            self.c = 227.9
        elif self.exchanger_type in ['flat_plate', 'flat plate']:
            self.a = 0.2581
            self.b = 891.7
            self.c = 26050  # 2.605×10^4
        else:
            raise ValueError("Invalid heat exchanger type. Please choose 'shell_and_tube' or 'flat_plate'.")
        
        # Dictionary of equations from Table A2.
        # Each equation is defined as a lambda function that computes the cost based on the area (A).
        self.table_a2 = {
            "a2a": lambda A: 10000 + 324 * (A ** 0.91),
            "a2b": lambda A: 130 * ((A / 0.093) ** 0.78),
            "a2c": lambda A: 16000 * ((A / 100) ** 0.6),
            "a2d": lambda A: 12000 * ((A / 100) ** 0.6),
            "a2e": lambda A: 1397 * (A ** 0.89),
            "a2f": lambda A: 2143 * (A ** 0.514),
            "a2g": lambda A: 2681 * (A ** 0.59),
            "a2h": lambda A: 96.2 * A,
            "a2i": lambda A: 34.9 * A,
            "a2j": lambda A: 45.7 * A,
            "a2k": lambda A: 309.14 * (A ** 0.85),
            "a2l": lambda A: 231.915 + (309.143 * A),
            "a2m": lambda A: 190 + (310 * A),
            "a2n": lambda A: 588 * (A ** 0.80),
            "a2o": lambda A: 7000 + (360 * (A ** 0.85)),
            "a2p": lambda A: 1.3 * (190 + (310 * A)),
            "a2q": lambda A: 3.28e4 * ((A / 80) ** 0.68),
            "a2r": lambda A: 383.5 * (A ** 0.65),
            "a2s": lambda A: 8500 + 409 * (A ** 0.85),
            "a2t": lambda A: 14000 + 614 * (A ** 0.92),
            "a2u": lambda A: 17500 + 699 * (A ** 0.93),
            "a2v": lambda A: 885 * (A ** 0.432)
        }
        # For flat plate exchangers, include additional equations.
        if self.exchanger_type in ['flat_plate', 'flat plate']:
            self.table_a2.update({
                "a2w": lambda A: 231 * (A ** 0.639),
                "a2x": lambda A: 1391 * (A ** 0.778),
                "a2y": lambda A: 635.14 * (A ** 0.778)
            })
    
    def compute_cost(self, area: float) -> float:
        """
        Computes the cost of the heat exchanger for a given area (in m²).
        
        If cost_method is "default", the general form is used:
          C = log(A) + a * A² + b * A + c
        Otherwise, the selected Table A2 equation is used.
        
        Parameters:
          area (float): Heat exchanger area in m²
          
        Returns:
          float: The calculated cost in dollars
        """
        if area <= 0:
            raise ValueError("Area must be a positive value.")
        
        if self.cost_method == "default":
            return math.log(area) + self.a * (area ** 2) + self.b * area + self.c
        elif self.cost_method in self.table_a2:
            return self.table_a2[self.cost_method](area)
        else:
            raise ValueError("The specified cost_method is not recognized. Please use 'default' or a valid Table A2 label.")
    
    def __str__(self):
        return (f"HeatExchangerCostModel(type='{self.exchanger_type}', "
                f"cost_method='{self.cost_method}')")


# #Example usage:
if __name__ == '__main__':
    area = 10.0  # Example area in m²
    
    # Using the default method (general form) for a Shell-and-Tube exchanger
    model_default = HeatExchangerCostModel("shell_and_tube", cost_method="default")
    cost_default = model_default.compute_cost(area)
    print(f"Cost using default method (Shell-and-Tube) for area = {area} m²: ${cost_default:,.2f}")
    
    # Using the default method (general form) for a flat_plate exchanger
    model_default = HeatExchangerCostModel("flat_plate", cost_method="default")
    cost_default = model_default.compute_cost(area)
    print(f"Cost using default method (flat_plate) for area = {area} m²: ${cost_default:,.2f}")
    
    # Using the Table A2 equation A2e for a Shell-and-Tube exchanger
    model_a2e = HeatExchangerCostModel("shell_and_tube", cost_method="a2e")
    cost_a2e = model_a2e.compute_cost(area)
    print(f"Cost using Table A2 method A2e (Shell-and-Tube) for area = {area} m²: ${cost_a2e:,.2f}")
    
    # Using the Table A2 equation A2x for a Flat Plate exchanger
    model_a2x = HeatExchangerCostModel("flat_plate", cost_method="a2x")
    cost_a2x = model_a2x.compute_cost(area)
    print(f"Cost using Table A2 method A2x (Flat Plate) for area = {area} m²: ${cost_a2x:,.2f}")


#%%
# import numpy as np
# import matplotlib.pyplot as plt

# # Generate 100 area values between 1 and 1000
# areas = np.linspace(1, 1000, 100)

# # Create a flat plate heat exchanger cost model using Table A2 equation A2x (as referenced in Figure 12)
# # model_flat = HeatExchangerCostModel("flat_plate", cost_method="default")
# model_flat = HeatExchangerCostModel("shell_and_tube", cost_method="default")

# # Compute cost for each area value
# costs = [model_flat.compute_cost(area) for area in areas]

# # Create a plot with equal scaling on both axes
# plt.figure(figsize=(8, 6))
# plt.plot(areas, costs)
# plt.xlabel("Area (m²)")
# plt.ylabel("Cost ($)")
# plt.title("Heat Exchanger Cost vs. Area")
# plt.legend()
# plt.grid(True)

# plt.show()
