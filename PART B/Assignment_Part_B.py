import math
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GRB, Model, quicksum

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

# Explicitly define the depot and customers:
depot = 0
C = [node for node in N if node != depot]  # Customers

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

# ======================================== CONSTRAINTS ========================================


# 1. Customer Visit Constraints (2) and (3)
# Each customer must be visited exactly once:
# - (2) ensures one outgoing route per customer node.
# - (3) ensures one incoming route per customer node.
model.addConstrs((quicksum(x[i, j, v] for j in N for v in V) == 1 for i in C),
                 name="EachCustomerVisitedOut")  # Equation (2)

model.addConstrs((quicksum(x[i, j, v] for i in N for v in V) == 1 for j in C),
                 name="EachCustomerVisitedIn")  # Equation (3)

# 2. Depot Constraints (4) and (5)
# Vehicles must start their route from the depot and must return to it:
# - (4) ensures each vehicle departs the depot exactly once.
# - (5) ensures each vehicle returns to the depot exactly once.
model.addConstrs((quicksum(x[depot, j, v] for j in N) == 1 for v in V),
                 name="StartAtDepot")   # Equation (4)

model.addConstrs((quicksum(x[i, depot, v] for i in N) == 1 for v in V),
                 name="ReturnToDepot")  # Equation (5)

# 3. Flow Conservation (6)
# For every node and vehicle, the inflow equals outflow:
# This ensures a continuous route without disjoint segments.
model.addConstrs(quicksum(x[i, j, v] for j in N) == quicksum(x[j, i, v] for j in N)
                 for i in N for v in V)  # Equation (6)

# 4. Time Window Constraints (7)
# Service must occur within each node’s time window:
# - Lower bound: t[j,v] >= earliest start time.
# - Upper bound: t[j,v] <= latest start time.
model.addConstrs((t[j, v] >= readytime[j] for j in N for v in V), "TimeWindow_LHS")     # Part of (7)
model.addConstrs((t[j, v] <= duetime[j] for j in N for v in V), "TimeWindow_RHS")       # Part of (7)

# Ensure travel and service times are respected in Big-M
longest_distance = np.max(c)
max_service = np.max(servicetime)
M = longest_distance + max_service + (np.max(duetime) - np.min(readytime))

# 5. Time Consistency (8)
# Ensure proper sequencing of service times and travel times:
# - t[j,v] >= t[i,v] + s_i + d[i,j] - M(1 - x[i,j,v]) for i in C, j in N
#   This ensures that the service start time at j does not occur before
#   arriving from i plus service time at i and travel time.
model.addConstrs(
    (t[j, v] >= t[i, v] + servicetime[i] + c[i, j] - M * (1 - x[i, j, v])
     for i in C for j in N for v in V), "TimeConsistencyCustomers"
)   # Equation (8) for customers

# Additional time consistency constraints for depot transitions:
# - Departing from depot to customer j:
model.addConstrs(
    (t[j, v] >= t_start[v] + servicetime[0] + c[0, j] - M * (1 - x[0, j, v])
     for j in C for v in V), "DepotTimeConsistency"
)   # Part of (8) for depot departure

# - Returning from customer i to depot:
model.addConstrs(
    (t_end[v] >= t[i, v] + servicetime[i] + c[i, 0] - M * (1 - x[i, 0, v])
     for i in C for v in V), "ReturnDepotTimeConsistency"
)   # Part of (8) for depot return

# 6. Depot-Specific Time Consistency
# The depot itself may also have a time window that must be respected:
model.addConstrs((t[depot, v] >= readytime[depot] for v in V),
                 "DepotStartTime_LHS")    # Ensures start time at depot is within window
model.addConstrs((t[depot, v] <= duetime[depot] for v in V),
                 "DepotStartTime_RHS")  # Ensures departure does not occur after the depot’s latest time

# 7. Vehicle Load Updates (9)
# Load progression along the route:
# - l[j,v] >= l[i,v] + D_j - M(1 - x[i,j,v]) ensures that if the vehicle travels from i to j,
#   the load at j is at least the load at i plus the demand of j.
# For transitions directly from the depot:
model.addConstrs(
  (l[j, v] >= l[i, v] + demand[j] - M * (1 - x[i, j, v])
   for i in C for j in C for v in V if i != j), "LoadUpdate_Customers"
)   # Equation (9) for customer-to-customer
model.addConstrs(
  (l[j, v] >= demand[j] - M * (1 - x[depot, j, v])
   for j in C for v in V), "LoadFromDepot")  # Equation (9) for depot-to-customer

# 8. Vehicle Capacity (10)
# Do not exceed vehicle capacity at any customer:
model.addConstrs((l[j, v] <= vehiclecapacity for j in C for v in V), "Capacity")    # Equation (10)

# 9. Depot Load (11)
# Set the vehicle load at the depot to zero at the start:
model.addConstrs((l[0, v] == 0 for v in V), "DepotLoad")  # Equation (11)

# 10. Subtour Elimination (12)
# Prevent formation of subtours using MTZ constraints:
# - u[i,v] - u[j,v] + n*x[i,j,v] <= n - 1 ensures that the sequence numbers (u) assigned
#   to visited nodes enforce a single continuous route rather than multiple small loops.
model.addConstrs(
  (u[i, v] - u[j, v] + len(C)*x[i, j, v] <= len(C) - 1
   for i in C for j in C if i != j for v in V), "Subtour"
)  # Equation (12)

# Binary and nonnegative variables (13) and (14)
# These are automatically defined by Gurobi during variable creation.

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
