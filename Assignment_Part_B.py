"""
Optimized Vehicle Routing Problem (VRP) model using Gurobi.

This script:
- Reads a small VRP instance from a data file.
- Sets up a mixed-integer programming model.
- Enforces route feasibility, time windows, vehicle capacity, and depot conditions.
- Optimizes the route selection to minimize total travel distance.
- Plots the resulting solution, color-coded per vehicle.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB, Model, quicksum
import matplotlib.cm as cm

# ============================================ MODEL DATA ============================================
# Read and parse data for the VRP problem
with open("data_small.txt", "r") as f:
    data = f.readlines()

VRP = np.array([[int(i) for i in line.split()] for line in data])

# Parameters
xc, yc = VRP[:, 1], VRP[:, 2]  # Coordinates of nodes
demand = VRP[:, 3]             # Demand at each node
servicetime = VRP[:, 4]        # Service time required at each node
readytime = VRP[:, 5]          # Earliest service start time
duetime = VRP[:, 6]            # Latest service start time

vehiclenumber = 2
vehiclecapacity = 130

# Sets and indices
N = VRP[:, 0]                  # Node indices
n = len(N)
V = range(vehiclenumber)       # Vehicle indices

# Distance matrix
c = np.array([[math.sqrt((xc[j] - xc[i]) ** 2 + (yc[j] - yc[i]) ** 2) for j in N] for i in N])

# ======================================== OPTIMIZATION MODEL ========================================
# Setup and solve the VRP model using Gurobi
model = Model('VRP_Model')

# Decision Variables
x = model.addVars(N, N, V, vtype=GRB.BINARY, name="x")
u = model.addVars(N, V, vtype=GRB.INTEGER, lb=0, name="u")
t = model.addVars(N, V, vtype=GRB.CONTINUOUS, lb=0, name="t")
l = model.addVars(N, V, vtype=GRB.CONTINUOUS, lb=0, name="l")
t_start = model.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="t_start")
t_end = model.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="t_end")
l_start = model.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="l_start")
l_end = model.addVars(V, vtype=GRB.CONTINUOUS, lb=0, name="l_end")

# Objective Function: minimize total travel distance
model.setObjective(quicksum(c[i, j] * x[i, j, v] for i in N for j in N for v in V), GRB.MINIMIZE)

# Constraints

# Each customer node is visited exactly once
model.addConstrs(quicksum(x[i, j, v] for j in N for v in V) == 1 for i in N[1:])
model.addConstrs(quicksum(x[i, j, v] for i in N for v in V) == 1 for j in N[1:])

# Vehicles start and end at the depot
model.addConstrs(quicksum(x[0, j, v] for j in N[1:]) == 1 for v in V)
model.addConstrs(quicksum(x[i, 0, v] for i in N[1:]) == 1 for v in V)

# Flow consistency
model.addConstrs(quicksum(x[i, j, v] for j in N) == quicksum(x[j, i, v] for j in N) for i in N for v in V)

# Time windows
model.addConstrs((t[j, v] >= readytime[j] for j in N for v in V), "TimeWindow_LHS")
model.addConstrs((t[j, v] <= duetime[j] for j in N for v in V), "TimeWindow_RHS")

# Big M for time consistency
M = max(c.flatten()) + max(servicetime) + max(duetime) - min(readytime)

model.addConstrs(
    (t[j, v] >= t[i, v] + servicetime[i] + c[i, j] - M * (1 - x[i, j, v])
     for i in N[1:] for j in N[1:] for v in V), "TimeConsistency"
)

# Depot-specific time consistency
model.addConstrs(
    (t_end[v] >= t_start[v] + quicksum(servicetime[j] * x[0, j, v] for j in N[1:]) for v in V), "DepotTime"
)
model.addConstrs((t_start[v] >= readytime[0] for v in V), "DepotStartTime_LHS")
model.addConstrs((t_start[v] <= duetime[0] for v in V), "DepotStartTime_RHS")

# Vehicle load update
model.addConstrs(
    (l[j, v] >= l[i, v] + demand[j] - M * (1 - x[i, j, v]) for i in N[1:] for j in N[1:] for v in V), "LoadUpdate"
)
model.addConstrs(
    (l[j, v] >= demand[j] - M * (1 - x[0, j, v]) for j in N[1:] for v in V), "LoadFromDepot"
)

# Vehicle capacity
model.addConstrs((l[j, v] <= vehiclecapacity for j in N for v in V), "Capacity")

# Depot load is zero
model.addConstrs((l[0, v] == 0 for v in V), "DepotLoad")

# Subtour elimination (MTZ constraints)
model.addConstrs(
    (u[i, v] - u[j, v] + n * x[i, j, v] <= n - 1
     for i in N for j in N if i != j and j != 0 for v in V), "Subtour"
)

# Solve the model
model.optimize()

# ========================================== SOLUTION PLOTTING =========================================
# Extract solution and plot routes
arc_solution = model.getAttr('x', x)

fig = plt.figure(figsize=(15, 15))
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title("Optimized VRP Routes")

# Plot depot and customers
plt.scatter(xc[1:], yc[1:], c='blue', label='Customers')
plt.scatter(xc[0], yc[0], c='red', label='Depot', marker='s')

# Annotate nodes
for i, (x_coord, y_coord) in enumerate(zip(xc, yc)):
    plt.annotate(f'{i}', (x_coord, y_coord))

# Predefined colors for vehicles
vehicle_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']

# Plot arcs
first_arc_plotted = [False] * vehiclenumber

for i in N:
    for j in N:
        for v in V:
            if arc_solution[i, j, v] > 0.5:
                color = vehicle_colors[v % len(vehicle_colors)]
                plt.plot([xc[i], xc[j]], [yc[i], yc[j]], color=color,
                         label=f"Vehicle {v}" if not first_arc_plotted[v] else "")
                first_arc_plotted[v] = True

plt.legend()
plt.show()

# ============================================ RESULTS ============================================
# Print results
for v in V:
    for i in N:
        print(f"Vehicle {v} visits node {i} at time {t[i, v].X:.2f}")

print(f"Objective value: {model.objVal}")