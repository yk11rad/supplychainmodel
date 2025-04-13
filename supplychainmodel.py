import pandas as pd
import pulp
import numpy as np
from scipy.stats import lognorm
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import os
import warnings
import gc
import psutil
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Log initial memory usage
print(f"Initial memory usage: {psutil.virtual_memory().percent}%")

# **Define Supply Chain Data**
suppliers = ['S1', 'S2', 'S3', 'S4']  # Four suppliers
warehouses = ['W1', 'W2', 'W3']       # Three warehouses
customers = ['C1', 'C2', 'C3', 'C4', 'C5']  # Five customers

# Supply capacity (units)
supply = {'S1': 500, 'S2': 400, 'S3': 450, 'S4': 550}

# Supplier disruption risks
supplier_risk = {'S1': 0.05, 'S2': 0.1, 'S3': 0.08, 'S4': 0.12}

# Warehouse data
warehouse_capacity = {'W1': 600, 'W2': 500, 'W3': 550}
holding_cost = {'W1': 0.8, 'W2': 0.9, 'W3': 1.0}
open_cost = {'W1': 1500, 'W2': 1600, 'W3': 1700}

# Stochastic demand scenarios
np.random.seed(42)
n_scenarios = 100
demand_scenarios = {
    'C1': [int(lognorm.rvs(s=0.3, scale=150)) for _ in range(n_scenarios)],
    'C2': [int(lognorm.rvs(s=0.25, scale=125)) for _ in range(n_scenarios)],
    'C3': [int(np.random.choice([90, 100, 110], p=[0.2, 0.6, 0.2])) for _ in range(n_scenarios)],
    'C4': [int(lognorm.rvs(s=0.32, scale=160)) for _ in range(n_scenarios)],
    'C5': [int(np.random.choice([70, 80, 90], p=[0.25, 0.5, 0.25])) for _ in range(n_scenarios)]
}
scenario_weights = [1/n_scenarios] * n_scenarios

# Check demand vs. supply
max_demands = {c: max(demand_scenarios[c]) for c in customers}
total_max_demand = sum(max_demands.values())
total_supply = sum(supply[s] * (1 - supplier_risk[s]) for s in suppliers)
print(f"Max demands: {max_demands}, Total: {total_max_demand}")
print(f"Total supply after risk: {total_supply}")

# Transportation costs (£/unit)
cost_s_w = {
    ('S1', 'W1'): 4, ('S1', 'W2'): 6, ('S1', 'W3'): 5,
    ('S2', 'W1'): 5, ('S2', 'W2'): 3, ('S2', 'W3'): 4,
    ('S3', 'W1'): 5, ('S3', 'W2'): 7, ('S3', 'W3'): 4,
    ('S4', 'W1'): 6, ('S4', 'W2'): 4, ('S4', 'W3'): 5
}
cost_w_c = {
    ('W1', 'C1'): 3, ('W1', 'C2'): 4, ('W1', 'C3'): 5, ('W1', 'C4'): 4, ('W1', 'C5'): 5,
    ('W2', 'C1'): 6, ('W2', 'C2'): 2, ('W2', 'C3'): 4, ('W2', 'C4'): 3, ('W2', 'C5'): 6,
    ('W3', 'C1'): 4, ('W3', 'C2'): 3, ('W3', 'C3'): 3, ('W3', 'C4'): 5, ('W3', 'C5'): 4
}

# Carbon emissions (kg CO2/unit)
carbon_s_w = {
    ('S1', 'W1'): 0.4, ('S1', 'W2'): 0.6, ('S1', 'W3'): 0.5,
    ('S2', 'W1'): 0.5, ('S2', 'W2'): 0.3, ('S2', 'W3'): 0.4,
    ('S3', 'W1'): 0.5, ('S3', 'W2'): 0.7, ('S3', 'W3'): 0.4,
    ('S4', 'W1'): 0.6, ('S4', 'W2'): 0.4, ('S4', 'W3'): 0.5
}
carbon_w_c = {
    ('W1', 'C1'): 0.3, ('W1', 'C2'): 0.4, ('W1', 'C3'): 0.5, ('W1', 'C4'): 0.4, ('W1', 'C5'): 0.5,
    ('W2', 'C1'): 0.6, ('W2', 'C2'): 0.2, ('W2', 'C3'): 0.4, ('W2', 'C4'): 0.3, ('W2', 'C5'): 0.6,
    ('W3', 'C1'): 0.4, ('W3', 'C2'): 0.3, ('W3', 'C3'): 0.3, ('W3', 'C4'): 0.5, ('W3', 'C5'): 0.4
}

# Emergency routes
emergency_routes = [('S1', 'C1'), ('S1', 'C4'), ('S2', 'C2'), ('S2', 'C5'), ('S3', 'C3'), ('S3', 'C4'), ('S4', 'C1'), ('S4', 'C5')]
emergency_cost = {('S1', 'C1'): 22, ('S1', 'C4'): 23, ('S2', 'C2'): 23, ('S2', 'C5'): 24, ('S3', 'C3'): 22, ('S3', 'C4'): 24, ('S4', 'C1'): 25, ('S4', 'C5'): 23}
emergency_carbon = {('S1', 'C1'): 1.5, ('S1', 'C4'): 1.6, ('S2', 'C2'): 1.2, ('S2', 'C5'): 1.3, ('S3', 'C3'): 1.4, ('S3', 'C4'): 1.5, ('S4', 'C1'): 1.6, ('S4', 'C5'): 1.4}

# **ML-based Demand Forecasting (Simulated)**
def forecast_demand():
    historical = pd.DataFrame({
        'features': np.random.rand(100),
        'demand': np.random.normal(250, 50, 100)
    })
    model = RandomForestRegressor(random_state=42)
    model.fit(historical[['features']], historical['demand'])
    return model.predict([[0.5]])[0]

# **Dynamic Cost Adjustment**
def optimize_routing(fuel_factor=1.0):
    return {(s, w): c * fuel_factor for (s, w), c in cost_s_w.items()}

# **Initialize DataFrames**
s_w_df = pd.DataFrame(columns=['Supplier', 'Warehouse', 'Units', 'Cost', 'Carbon'])
w_c_df = pd.DataFrame(columns=['Warehouse', 'Customer', 'Units', 'Cost', 'Carbon'])
z_df = pd.DataFrame(columns=['Supplier', 'Customer', 'Units', 'Cost', 'Carbon'])
inventory_df = pd.DataFrame(columns=['Warehouse', 'Inventory', 'Holding_Cost', 'Open'])
total_s_w_cost = total_w_c_cost = total_z_cost = total_holding_cost = open_cost_total = total_carbon = total_cost = 0

# **Initialize PuLP Model**
model = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)

# **Decision Variables**
x = pulp.LpVariable.dicts("x", [(s, w, i) for s in suppliers for w in warehouses for i in range(n_scenarios)], lowBound=0, cat='Continuous')
y = pulp.LpVariable.dicts("y", [(w, c, i) for w in warehouses for c in customers for i in range(n_scenarios)], lowBound=0, cat='Continuous')
inv = pulp.LpVariable.dicts("inv", [(w, i) for w in warehouses for i in range(n_scenarios)], lowBound=0, cat='Continuous')
z = pulp.LpVariable.dicts("z", [(s, c, i) for (s, c) in emergency_routes for i in range(n_scenarios)], lowBound=0, cat='Continuous')
open_w = pulp.LpVariable.dicts("Open", warehouses, cat='Binary')

# **Dynamic Costs**
fuel_factor = 1.1
dynamic_cost_s_w = optimize_routing(fuel_factor)

# **Objective Function**
model += (
    pulp.lpSum(
        scenario_weights[i] * (
            pulp.lpSum(dynamic_cost_s_w.get((s, w), cost_s_w[(s, w)]) * x[(s, w, i)] for s in suppliers for w in warehouses) +
            pulp.lpSum(cost_w_c[(w, c)] * y[(w, c, i)] for w in warehouses for c in customers) +
            pulp.lpSum(emergency_cost[(s, c)] * z[(s, c, i)] for (s, c) in emergency_routes) +
            pulp.lpSum(holding_cost[w] * inv[(w, i)] for w in warehouses)
        ) for i in range(n_scenarios)
    ) + pulp.lpSum(open_cost[w] * open_w[w] for w in warehouses),
    "Total_Expected_Cost"
)

# **Constraints**
# Supply limits
for s in suppliers:
    for i in range(n_scenarios):
        model += (
            pulp.lpSum(x[(s, w, i)] for w in warehouses) +
            pulp.lpSum(z[(s, c, i)] for (s, c) in emergency_routes if s == s) <=
            supply[s] * (1 - supplier_risk[s]),
            f"Supply_{s}_{i}"
        )

# Demand satisfaction
for c in customers:
    for i in range(n_scenarios):
        model += (
            pulp.lpSum(y[(w, c, i)] for w in warehouses) +
            pulp.lpSum(z[(s, c, i)] for (s, c) in emergency_routes if c == c) >=
            demand_scenarios[c][i],
            f"Demand_{c}_{i}"
        )

# Warehouse capacity
for w in warehouses:
    for i in range(n_scenarios):
        model += (
            inv[(w, i)] <= warehouse_capacity[w] * open_w[w],
            f"Capacity_{w}_{i}"
        )
        model += (
            pulp.lpSum(x[(s, w, i)] for s in suppliers) <=
            warehouse_capacity[w] * open_w[w],
            f"Input_Capacity_{w}_{i}"
        )

# Flow balance
for w in warehouses:
    for i in range(n_scenarios):
        model += (
            pulp.lpSum(x[(s, w, i)] for s in suppliers) ==
            pulp.lpSum(y[(w, c, i)] for c in customers) + inv[(w, i)],
            f"Flow_Balance_{w}_{i}"
        )

# Minimum one warehouse open
model += pulp.lpSum(open_w[w] for w in warehouses) >= 1, "Min_One_Warehouse"

# Carbon constraint
carbon_budget = 2000
model += (
    pulp.lpSum(
        scenario_weights[i] * (
            pulp.lpSum(carbon_s_w[(s, w)] * x[(s, w, i)] for s in suppliers for w in warehouses) +
            pulp.lpSum(carbon_w_c[(w, c)] * y[(w, c, i)] for w in warehouses for c in customers) +
            pulp.lpSum(emergency_carbon[(s, c)] * z[(s, c, i)] for (s, c) in emergency_routes)
        ) for i in range(n_scenarios)
    ) <= carbon_budget,
    "Carbon_Limit"
)

# Capital budget
capital_budget = 20000
model += (
    pulp.lpSum(open_cost[w] * open_w[w] for w in warehouses) <= capital_budget,
    "Capital_Limit"
)

# Emergency route limit (20% of average demand)
average_total_demand = np.mean([sum(demand_scenarios[c][i] for c in customers) for i in range(n_scenarios)])
model += (
    pulp.lpSum(z[(s, c, i)] for (s, c) in emergency_routes for i in range(n_scenarios)) <= 0.2 * average_total_demand * n_scenarios,
    "Emergency_Limit"
)

# **Solve Model**
model.solve()

# **Process Results**
if pulp.LpStatus[model.status] == 'Optimal':
    print("Optimal solution found!")
    
    # Supplier-to-warehouse flows
    s_w_flows = []
    for s in suppliers:
        for w in warehouses:
            flow = np.mean([x[(s, w, i)].varValue or 0 for i in range(n_scenarios)])
            if flow > 1e-6:
                s_w_flows.append({
                    'Supplier': s,
                    'Warehouse': w,
                    'Units': flow,
                    'Cost': flow * dynamic_cost_s_w.get((s, w), cost_s_w[(s, w)]),
                    'Carbon': flow * carbon_s_w[(s, w)]
                })

    # Warehouse-to-customer flows
    w_c_flows = []
    for w in warehouses:
        for c in customers:
            flow = np.mean([y[(w, c, i)].varValue or 0 for i in range(n_scenarios)])
            if flow > 1e-6:
                w_c_flows.append({
                    'Warehouse': w,
                    'Customer': c,
                    'Units': flow,
                    'Cost': flow * cost_w_c[(w, c)],
                    'Carbon': flow * carbon_w_c[(w, c)]
                })

    # Emergency flows
    z_flows = []
    for (s, c) in emergency_routes:
        flow = np.mean([z[(s, c, i)].varValue or 0 for i in range(n_scenarios)])
        if flow > 1e-6:
            z_flows.append({
                'Supplier': s,
                'Customer': c,
                'Units': flow,
                'Cost': flow * emergency_cost[(s, c)],
                'Carbon': flow * emergency_carbon[(s, c)]
            })

    # Inventory and warehouse status
    inventory = []
    for w in warehouses:
        level = np.mean([inv[(w, i)].varValue or 0 for i in range(n_scenarios)])
        inventory.append({
            'Warehouse': w,
            'Inventory': level,
            'Holding_Cost': level * holding_cost[w] if level > 0 else 0,
            'Open': open_w[w].varValue
        })

    # Update DataFrames
    s_w_df = pd.DataFrame(s_w_flows)
    w_c_df = pd.DataFrame(w_c_flows)
    z_df = pd.DataFrame(z_flows)
    inventory_df = pd.DataFrame(inventory)

    # Calculate totals
    total_s_w_cost = s_w_df['Cost'].sum() if not s_w_df.empty else 0
    total_w_c_cost = w_c_df['Cost'].sum() if not w_c_df.empty else 0
    total_z_cost = z_df['Cost'].sum() if not z_df.empty else 0
    total_holding_cost = inventory_df['Holding_Cost'].sum() if not inventory_df.empty else 0
    open_cost_total = sum(open_cost[w] * open_w[w].varValue for w in warehouses)
    total_cost = total_s_w_cost + total_w_c_cost + total_z_cost + total_holding_cost + open_cost_total

    total_carbon = (
        (s_w_df['Carbon'].sum() if not s_w_df.empty else 0) +
        (w_c_df['Carbon'].sum() if not w_c_df.empty else 0) +
        (z_df['Carbon'].sum() if not z_df.empty else 0)
    )

    # Print results
    print("\nSupplier to Warehouse Flows (Average):")
    print(s_w_df if not s_w_df.empty else "No flows.")
    print("\nWarehouse to Customer Flows (Average):")
    print(w_c_df if not w_c_df.empty else "No flows.")
    print("\nEmergency Flows (Average):")
    print(z_df if not z_df.empty else "No emergency flows.")
    print("\nWarehouse Inventory and Status:")
    print(inventory_df)
    print("\nCost and Carbon Summary:")
    print(f"Supplier to Warehouse Cost: £{total_s_w_cost:.2f}")
    print(f"Warehouse to Customer Cost: £{total_w_c_cost:.2f}")
    print(f"Emergency Transport Cost: £{total_z_cost:.2f}")
    print(f"Inventory Holding Cost: £{total_holding_cost:.2f}")
    print(f"Warehouse Opening Cost: £{open_cost_total:.2f}")
    print(f"Total Cost: £{total_cost:.2f}")
    print(f"Total Carbon Emissions: {total_carbon:.2f} kg CO2")
else:
    print(f"Solver status: {pulp.LpStatus[model.status]}")

# Free memory
gc.collect()

# **Save to CSV**
output_file = 'supply_chain_results.csv'
try:
    with open(output_file, 'w') as f:
        if not s_w_df.empty:
            s_w_df.assign(Flow_Type='Supplier_to_Warehouse').to_csv(
                f, index=False, mode='w', header=True
            )
        if not w_c_df.empty:
            w_c_df.assign(Flow_Type='Warehouse_to_Customer').to_csv(
                f, index=False, mode='a', header=not f.tell()
            )
        if not z_df.empty:
            z_df.assign(Flow_Type='Emergency', Warehouse=np.nan).to_csv(
                f, index=False, mode='a', header=not f.tell()
            )
        if not inventory_df.empty:
            inventory_df.assign(Flow_Type='Inventory').rename(
                columns={'Inventory': 'Units', 'Holding_Cost': 'Cost'}
            ).to_csv(f, index=False, mode='a', header=not f.tell())
    print(f"\nResults saved to '{output_file}'")
    files.download(output_file)
except Exception as e:
    print(f"Error saving CSV: {e}")

# **Visualizations**
flows_df = pd.DataFrame()
if not s_w_df.empty or not w_c_df.empty:
    flows_list = []
    if not s_w_df.empty:
        flows_list.append(
            s_w_df.assign(Route=[f"{r['Supplier']}→{r['Warehouse']}" for _, r in s_w_df.iterrows()])
        )
    if not w_c_df.empty:
        flows_list.append(
            w_c_df.assign(Route=[f"{r['Warehouse']}→{r['Customer']}" for _, r in w_c_df.iterrows()])
        )
    flows_df = pd.concat(flows_list, ignore_index=True)

if not flows_df.empty:
    coords = {
        'S1': (51.5, -0.1), 'S2': (53.4, -2.2), 'S3': (52.0, -1.0), 'S4': (54.0, -3.0),
        'W1': (52.5, -1.9), 'W2': (51.4, -2.6), 'W3': (54.5, -1.5),
        'C1': (50.8, -1.0), 'C2': (53.8, -1.5), 'C3': (51.9, -2.1), 'C4': (52.3, -0.5), 'C5': (53.0, -2.5)
    }
    flows_df['latitude'] = flows_df.apply(
        lambda r: coords.get(r['Supplier'], coords.get(r['Warehouse']))[0], axis=1
    )
    flows_df['longitude'] = flows_df.apply(
        lambda r: coords.get(r['Supplier'], coords.get(r['Warehouse']))[1], axis=1
    )
    fig = px.scatter_geo(
        flows_df,
        lat='latitude',
        lon='longitude',
        size='Units',
        color='Cost',
        hover_name='Route',
        scope='europe',
        title='Supply Chain Network'
    )
    fig.update_geos(fitbounds="locations")
    fig.show()

cost_data = {
    'Category': ['S-to-W', 'W-to-C', 'Emergency', 'Inventory', 'Opening'],
    'Cost': [total_s_w_cost, total_w_c_cost, total_z_cost, total_holding_cost, open_cost_total]
}
cost_df = pd.DataFrame(cost_data)
plt.figure(figsize=(10, 5))
sns.barplot(x='Category', y='Cost', data=cost_df)
plt.title('Cost Breakdown')
plt.ylabel('Cost (£)')
plt.show()

# Free memory
gc.collect()