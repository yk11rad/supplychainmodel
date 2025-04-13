# supplychainmodel
Overview
This Google Colab notebook implements a Supply Chain Optimization Model for a multi-tier supply chain involving suppliers, warehouses, and customers. The model uses PuLP for linear programming to minimize total costs while satisfying demand, capacity, and carbon constraints. It incorporates stochastic demand scenarios, supplier disruption risks, emergency routes, and dynamic cost adjustments. Additionally, it includes a simulated RandomForestRegressor for demand forecasting and generates interactive visualizations using Plotly and Matplotlib.

The supply chain consists of:

4 Suppliers (S1–S4) with varying capacities and disruption risks.
3 Warehouses (W1–W3) with capacity, holding, and opening costs.
5 Customers (C1–C5) with stochastic demand across 100 scenarios.
Emergency Routes for direct supplier-to-customer shipments at higher costs/emissions.
The model optimizes flows, inventory levels, and warehouse openings to balance cost (transport, holding, opening) and carbon emissions, subject to a carbon budget and capital constraints.

Key Features
Stochastic Demand: Models customer demand using lognormal and discrete distributions across 100 scenarios.
Risk Modeling: Accounts for supplier disruption risks (5–12% chance of reduced capacity).
Optimization:
Minimizes total expected cost (transport, inventory, warehouse opening).
Enforces constraints: supply limits, demand satisfaction, warehouse capacity, flow balance, carbon budget (2,000 kg CO2), capital budget (£20,000), and emergency route limits (20% of average demand).
Dynamic Costs: Adjusts supplier-to-warehouse transport costs with a fuel factor (default: 1.1).
Emergency Routes: Allows high-cost, high-emission direct shipments for flexibility.
ML Forecasting: Simulates demand forecasting with a RandomForestRegressor (placeholder implementation).
Outputs:
Flow tables: Supplier-to-warehouse, warehouse-to-customer, and emergency flows.
Inventory and warehouse status (open/closed, holding costs).
Cost breakdown: Transport, inventory, opening.
Carbon emissions summary.
CSV export: supply_chain_results.csv with all flows and inventory data.
Visualizations:
Interactive geographic scatter plot (Plotly) showing flow routes and units.
Bar plot (Matplotlib) of cost categories.
Performance:
Memory management with gc.collect() and usage logging via psutil.
Warning suppression for cleaner output.
Requirements
Environment: Google Colab (script uses google.colab.files for CSV download).
Libraries:
pandas, numpy, pulp, scipy, scikit-learn, plotly, matplotlib, seaborn, psutil, shutil.
Install with: !pip install pandas pulp numpy scipy scikit-learn plotly matplotlib seaborn psutil.
No external data files required; all inputs are defined in the script.
