
#imports
import operator
import geopandas as gpd
import random
import matplotlib.pyplot as plt
from functools import partial
import networkx as nx
from gerrychain import MarkovChain
from gerrychain.constraints import (
    Validator,
    single_flip_contiguous,
    within_percent_of_ideal_population,
)
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election, Tally, cut_edges
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
import csv
import os
from functools import partial
import json
from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.tree import recursive_tree_part
import numpy as np
from sklearn import manifold
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from itertools import combinations, product
import pickle
import wasserplan
from scipy.optimize import linear_sum_assignment


graph = Graph.from_json("./NC_VTDjson.json")
pop_col = "TOTPOP"
updaters1 = {
    "population": updaters.Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
}
total_population = sum([graph.nodes[n][pop_col] for n in graph.nodes()])
initial_partition = Partition(graph, "newplan", updaters1)
ideal_population = sum(initial_partition["population"].values())/len(initial_partition)
compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
)
proposal = partial(recom,
                   pop_col=pop_col,
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )
chain = MarkovChain(
    proposal=proposal,
    constraints=[
        constraints.within_percent_of_ideal_population(initial_partition, 0.02),
        compactness_bound,
    ],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=12000
)
parts = []
for index, part in enumerate(chain):
    if ((index % 100 == 0) and (index > 1000)):
        print(index)
        parts.append(part)
print(" ")
print(len(parts), " parts stored.")


import wasserplan

def reindexto1(part):
    to_1index = {x:(index+1) for index, x in enumerate(part.parts)}
    ass = {x:to_1index[part.assignment[x]] for x in graph.nodes()}
    return Partition(part.graph, ass, part.updaters)

def getpart(i):
    humanlist = ['judge', 'oldplan', 'newplan']
    if i < 3:
        return reindexto1(Partition(graph, humanlist[i], updaters1))
    else:
        return reindexto1(parts[i])
def distance(p0,p1):
    pair = wasserplan.Pair(p0, p1)
    return pair.distance

M = np.zeros((100, 100))
for i in range(100):
    print(i, end=" ")
    for j in range(i, 100):
        M[i][j] = distance(getpart(i), getpart(j))

pickle.dump(M, open("NCtransMat.p", "wb"))

#plot
M = M + M.T
from sklearn.manifold import MDS
mds = MDS(n_components=2, dissimilarity='precomputed', max_iter=50000, n_init=100)
pos=mds.fit(M).embedding_
X=[]
Y=[]
for i in range(100):
    X.append(pos[i][0])
    Y.append(pos[i][1])

plt.scatter(X,Y)
plt.scatter(X[:3], Y[:3], color='red')
plt.annotate("judge", (X[0],Y[0]))
plt.annotate("2012", (X[1],Y[1]))
plt.annotate("2016", (X[2],Y[2]))
plt.savefig("NCtransplot.png")
