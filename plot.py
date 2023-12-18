import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


folder_path = "./results_euler/"

type = "homophilic"
# type = "backbone"
edges = 20
pf = 0.8
pff = 0.1
mu = 0.9
theta = 0.1
exp = 1


file = type+"_N100_μ" + str(mu) + "_θ" + str(theta) + "_E" + str(edges)
if edges == 7:
    file = file + ".000000000000001"
if edges == 13:
    file = file + ".0"
if edges == 20:
    file = file + ".0"
if type == "homophilic":
    file = file + "_pf" + str(pf) + "_pff" + str(pff)
file = file + "_" + str(exp)

data = pd.read_csv(folder_path+file+".csv").iloc[:,1:]

n_step = data.shape[0]
n_agents = data.shape[1]

X = []
Y = []
for i in range(n_step):
    for j in range(n_agents):
        X.append(i)
        Y.append(data.iloc[i,j])

fig, ax = plt.subplots()
scatter = ax.scatter(X, Y, c='k', s=1, edgecolors="none")

plt.title(file)
plt.xlabel("Step")
plt.ylabel("Opinion")
plt.savefig("./time_plot/"+ file +".png")





