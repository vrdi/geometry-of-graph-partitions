import os
import random
import functools
import numpy as np
import math
import networkx as nx
import numpy as np
from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election,Tally,cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part
from LMDS import landmarkMDS_2D as lmds
import wasserplan
import pickle

base=2
pop_tol=.05
STEPS = 500000
INTERVAL = 10000

def step_num(partition):
    parent = partition.parent
    if not parent:
        return 0
    return parent["step_num"] + 1

def annealing_cut_accept2(partition):
    boundaries1 = {x[0] for x in partition["cut_edges"]}.union(
        {x[1] for x in partition["cut_edges"]}
    )
    boundaries2 = {x[0] for x in partition.parent["cut_edges"]}.union(
        {x[1] for x in partition.parent["cut_edges"]}
    )
    t = partition["step_num"]
    if t < int(STEPS/5):
        beta = 0
    elif t < int(0.8*STEPS):
        beta = (t - int(STEPS/5)) / int(STEPS/5)
    else:
        beta = 3
    bound = 1
    if partition.parent is not None:
        exponent = beta * (
            -len(partition["cut_edges"]) + len(partition.parent["cut_edges"])
        )
        bound = (base ** (exponent)) * (len(boundaries1) / len(boundaries2))
    return random.random() < bound

def getbeta(t):
    if t < int(STEPS/5):
        beta = 0
    elif t < int(0.8*STEPS):
        beta = (t - int(STEPS/5)) / int(STEPS/5)
    else:
        beta = 3
    return beta

fips="05"
graph = Graph.from_json("./BG05/BG05.json")
totpop = sum([int(graph.nodes[n]["TOTPOP"]) for n in graph.nodes])

for n in graph.nodes:
    graph.nodes[n]["TOTPOP"] = int(graph.nodes[n]["TOTPOP"])

betas = []
ts = []
myupdaters = {
    'population': Tally('TOTPOP', alias="population"),
    'cut_edges': cut_edges,
    'step_num': step_num,
}

runlist = [0]
partdict = {r:[] for r in runlist}
allparts = []

#run annealing flip
for run in runlist:
    initial_ass =  recursive_tree_part(graph, range(6), totpop/6, "TOTPOP", .01, 1)
    initial_partition = Partition(graph,assignment=initial_ass,updaters=myupdaters)
    popbound=within_percent_of_ideal_population(initial_partition,pop_tol)
    ideal_population = totpop / 6

    print("Dumping seed", run)
    pickle.dump(initial_ass, open("oneseed"+str(run), "wb"))

    #make flip chain
    exp_chain = MarkovChain(
        propose_random_flip,
        constraints = [single_flip_contiguous,popbound],
        accept = annealing_cut_accept2,
        initial_state=initial_partition,
        total_steps = STEPS
    )
    for index, part in enumerate(exp_chain):
        if index%INTERVAL == 0:
            print(run,"-",index)
            allparts.append(part)
            partdict[run].append(part)
        if ((index+1) in [int(z*STEPS/5) for z in range(0,6)]):
            print("saving assignment", index)
            pickle.dump(part.assignment, open("onerun"+str(run)+"at"+str(index), "wb"))

def getpart(i):
    return allparts[i]
def transport(part0, part1):
    pair = wasserplan.Pair(part0, part1)
    return pair.distance

X,Y = lmds(
    getpart,
    len(allparts),
    transport,
    verbose=True,
    plot=True
)

pickle.dump((X,Y), open("annealoneXY.p", "wb"))
