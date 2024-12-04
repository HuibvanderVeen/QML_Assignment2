#================================================================================
#Bilge Atasoy
#==================================================================================================


from gurobipy import *
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('text', usetex=True)
#import os
#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

#============================================MODEL DATA============================================


with open("/Users/huibvanderveen/Desktop/Pycharm Projects/TU/QML/dataTSPexample.txt") as f:          # Open Li & Lim PDPTW instance definitions
    data = f.readlines()                        # Extract instance definitions

TSP = []                                        # Create array for data related to nodes
i=0                                             # Varible to keep track of lines in data file
for line in data:
    i=i+1
    words = line.split()
    words=[int(i) for i in words]           # Covert data from string to integer
    TSP.append(words)                       # Store node data
TSP = np.array(TSP)


V=TSP[:,0]                                      # Nodes
n=len(V)                                        # Number of nodes

xc=TSP[:,1]                                     # X-position of nodes
yc=TSP[:,2]                                     # Y-position of nodes
# the remainder of the data file is not used yet 


t=np.zeros((n,n))                               # Create array for distance between nodes
for i in V:
    for j in V:
        t[i][j]=math.sqrt((xc[j] - xc[i])**2 + (yc[j] - yc[i])**2) # Store distance between nodes


#========================================OPTIMIZATION MODEL========================================


## Create optimization model
m = Model('TSPmodel')

## Create Decision Variables
#arcs - if  arc(i,j) is visited it is 1
x = {}
for i in V:
    for j in V:
        x[i,j] = m.addVar(vtype=GRB.BINARY, lb = 0, name="X_%s,%s" %(i,j))
      
#order for subtour elimination
u = {}
for i in V:
    u[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="W_%s" %(i))        

## Objective
obj = (quicksum(t[i,j]*x[i,j] for i in V for j in V))
m.setObjective(obj, GRB.MINIMIZE)

# Constraints    
# All locations should be visited
for i in V:
    m.addConstr(quicksum(x[i,j] for j in V) == 1, name='Visit_%s' % (i))
for j in V:
    m.addConstr(quicksum(x[i,j] for i in V) == 1, name='Visit_%s' % (i))
       

# #SUBTOUR ELIMINATION     
# ##maximum and minimum values for each node except the start  
m.addConstr(u[0] == 1) 
for i in V: 
    if i != 0:
        m.addConstr(u[i] >= 2)
        m.addConstr(u[i] <= len(V))
#the order of nodes based on the decision variables  
for i in V:
    for j in V:
        if j!=0:
            m.addConstr(u[i] - u[j] + (len(V)) * x[i,j] <= len(V)-1)       
    
# # No connection to the node itself
# for i in V:
#     m.addConstr(x[i,i],GRB.EQUAL,0,name='NoC_%s' % (i)) 
    
 
m.update()
m.write('TSPmodel.lp')
m.Params.timeLimit = 3600
m.optimize()
m.write('TSPmodel.sol')

# Plot the routes that are decided to be traversed 
arc_solution = m.getAttr('x', x)
#
fig= plt.figure(figsize=(15,15))
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.scatter(xc[1:n-1],yc[1:n-1])
for i in range(1,n-1):
    plt.annotate(str(i),(xc[i],yc[i]))
plt.plot(xc[0],yc[0],c='g',marker='s')
#
for i in range(n):
    for j in range(n):
        if arc_solution[i,j] > 0.99:
            plt.plot([xc[i], xc[j]], [yc[i], yc[j]],'r--')
plt.show()          
##YOU CAN SAVE YOUR PLOTS SOMEWHERE IF YOU LIKE
##plt.savefig('Plots/TSP.png',bbox_inches='tight')   
#

for i in V:
    print("ORDER OF NODE " , i, " IS " , u[i].X)
                           

print('Obj: %g' % m.objVal)

Totaldistance = sum(t[i,j]*x[i,j].X for i in V for j in V)

print('Total distance traveled: ', Totaldistance)


