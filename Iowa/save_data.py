#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
from collections import defaultdict 

def dump_run(filename, partitions):
    """Dumps a chain run."""
    initial_assignment = defaultdict(list)
    for precinct_idx, district_idx in partitions[0].assignment.items():
        initial_assignment[district_idx].append(precinct_idx)
    
    run_serialized = {
        'initial': initial_assignment,
        'deltas': []
    }
    last_plan = partitions[0]
    for plan in partitions[1:]:
        delta = defaultdict(list)
        for precinct_idx, district_idx in plan.assignment.items():
            if district_idx != last_plan.assignment[precinct_idx]:
                delta[district_idx].append(precinct_idx)
        run_serialized['deltas'].append(delta)
        last_plan = plan
    with open(filename, 'w') as f:
        json.dump(run_serialized, f)
    
        
def load_run(filename, initial_partition):
    """Loads a chain run.
    
    :param filename: The name of a chain file dumped by :func:`dump_run`.
    :param initial_partition: A ``gerrychain.Partition`` with the same
        graph and updaters as the partition diffs in the chain file.
    """
    partition_diffs = json.load(open(filename))
    last_partition = initial_partition.flip(remap(partition_diffs['initial']))
    partitions = [last_partition]
    for delta in partition_diffs['deltas']:
        last_partition = last_partition.flip(remap(delta))
        partitions.append(last_partition)
    return partitions
    
        
def remap(diff):
    """Changes the format of a chain diff.
    
    :param diff: A diff in the format {district: [node 1, node 2, ...]}
    :returns: A diff in the format {node 1: district, node 2: district, ...}
    """
    flipped = {}
    for district, nodes in diff.items():
        for node in nodes:
            flipped[node] = district
    return flipped
