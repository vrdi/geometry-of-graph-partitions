#imports
import operator
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
import pickle
import wasserplan

graph = Graph.from_json("NC_VTD.json")
def get_partition(partition_file, fid_to_geoid_file, geoidcol):
    fid_to_node = get_fid_to_node(fid_to_geoid_file, geoidcol)
    fid_to_district = {}
    f = open(partition_file, 'r')
    for line in f:
        lineitems = line.rstrip("\n").split("\t")
        fid, district = int(lineitems[0])+1, int(lineitems[1])
        fid_to_district[fid] = district
    assignment = {}
    for fid in fid_to_district:
        assignment[fid_to_node[fid]] = fid_to_district[fid]-1 #make 0-indexed
    return assignment
def get_fid_to_node(filename, geoidcol):
    geoid_to_fid = {}
    fid_to_node = {}
    f = open(filename, 'r')
    f.readline()
    for line in f:
        lineitems = line.rstrip("\n").split("\t")
        fid, geoid = int(lineitems[0]), lineitems[1]
        geoid_to_fid[geoid] = fid
    for n in graph.nodes:
        fid_to_node[geoid_to_fid[graph.nodes[n][geoidcol]]] = n
    return fid_to_node

#set up geography and elections
pop_col = "TOTPOP"
my_updaters = {
    "population": updaters.Tally(pop_col, alias="population"),
    "cut_edges": cut_edges,
}


import os
folder = "districtMaps"
fid_to_geoid_file = "./preprocessedData/arcgis_FIDtovtdkeymap.txt"
geoidcol = "VTD"
files = []
pop_col = "TOTPOP"
for filename in os.listdir(folder):
    #print(filename)
    files.append("./"+folder+"/"+filename)

def getpart(i):
    humanlist = ['judge', 'oldplan', 'newplan']
    if i < 3:
        ass = Partition(graph, humanlist[i], my_updaters).assignment
        to0index = {x:i for i, x in enumerate(ass.parts)}
        return Partition(graph, {n:to0index[ass[n]] for n in graph.nodes}, my_updaters)
    else:
        ass = get_partition(files[240*(i)],fid_to_geoid_file, geoidcol)
        return Partition(graph, ass, my_updaters)

def distance(p0,p1):
    pair = wasserplan.Pair(p0, p1)
    return pair.distance

M = np.zeros((100, 100))
for i in range(100):
    for j in range(i, 100):
        print(i, "to", j)
        M[i][j] = distance(getpart(i), getpart(j))
    pickle.dump(M, open("NCMattinglyMat.p", "wb"))

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
plt.savefig("NCMattinglyplot.png")
