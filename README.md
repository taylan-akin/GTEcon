# GTEcon



**Python library for techno-economic assessment of high‑temperature geothermal ATES, BTES, and MTES systems.**

---

<p align="center">
  <img src="docs/logo/Logo-push-it-final_horizontaal-1024x446.png" alt="PUSH-IT Logo" width="400" />
</p>

This work was developed by **Dr. Taylan AKIN** (takin@pau.edu.tr, t.akin@tudelft.nl) within the EU-funded **PUSH-IT** project ([https://www.push-it-thermalstorage.eu/](https://www.push-it-thermalstorage.eu/)) in collaboration with **Dr. Alexandros DANIILIDIS** ([A.Daniilidis@tudelft.nl](mailto\:A.Daniilidis@tudelft.nl)).

GTEcon is a Python library that transforms pandas DataFrames of well-operation time-series into a full techno-economic assessment tool for high-temperature geothermal energy storage. It supports Aquifer (ATES), Borehole (BTES) and Mine (MTES) Thermal Energy Storage systems by automating calculations of pumping and heat-production power, cash flows, Net Present Value (NPV), Levelized Cost of Heat (LCOH) and payback periods. Inputs can be loaded directly from structured Excel workbooks or defined programmatically, and built-in Monte Carlo and sensitivity analyses enable robust uncertainty quantification. Whether you’re exploring project feasibility interactively or integrating into batch simulations, GTEcon provides a reproducible, flexible framework for evaluating the economic viability of geothermal storage projects

## 🚀 Features

- **Pandas extension**: attach `.gte` accessor to any `DataFrame` containing well-operation time‑series.
- **Automatic input loading**: read units, well parameters, cost components, and economic settings from an Excel workbook.
- **Manual configuration**: assign inputs programmatically via Python when you need fine control.
- **Comprehensive analysis**: compute heat/pump power, cash flows, NPV, LCOH, and payback periods.
- **Monte Carlo simulation**: perform uncertainty analysis on discount rates, heat prices, and other key parameters.

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/taylan-akin/GTEcon.git
   cd GTEcon
   ```

2. **Create a virtual environment (optional, recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .\.venv\Scripts\activate  # Windows PowerShell
   ```

## 📖 Usage

### 1. Automatic input loading

```python
from GTEcon_module import GTEcon

econ = GTEcon(
    well_operation_path="path/to/well_data.xlsx",
    params_path="path/to/parameters.xlsx"
)
econ.run(plotting=True)
print(f"LCOH: {econ.lcoe:.2f} €/MWh\nNPV: {econ.npv:.2f} €")
```

### 2. Manual configuration

```python
from GTEcon_module import GTEcon, units
from hx import HeatExchangerCostModel

econ = GTEcon(well_operation_path="path/to/well_data.xlsx")

# Assign unit conversions and well parameters
econ.well_inputs.time_column_name = 'Time (yrs)'
econ.well_inputs.CoEff_units     = units("Day", "Kelvin", "Bar", "m3/day")
econ.well_inputs.well_names      = ['H1', 'W1']
# ... assign other well_inputs and econ_inputs ...

econ.add_cost(
    cost_item="Heat Exchanger",
    unit_cost=HeatExchangerCostModel("flat_plate").compute_cost(10),
    quantity=1,
    repl_freq=0
)

# Run analysis
econ.run()
print(f"LCOH: {econ.lcoe:.2f} €/MWh, NPV: {econ.npv:.2f} €")
```

For full examples, see [run_GTEcon.py](run_GTEcon.py).

## 🎓 Documentation

Additional details and API reference are available in the `docs/` folder (coming soon).

## ⚠️ Disclaimer
This work was funded by the European Union under the Horizon Europe programme (grant no. 1011096566). Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or CINEA. Neither the European Union nor CINEA can be held responsible for them.

## 🛡 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request with bug fixes, enhancements, or documentation improvements.

## 📧 Contact

Taylan Akın: [takin@pau.edu.tr](mailto\:takin@pau.edu.tr), [t.akin@tudelft.nl](mailto\:t.akin@tudelft.nl)\
Project link: [https://github.com/](https://github.com/)/GTEcon

