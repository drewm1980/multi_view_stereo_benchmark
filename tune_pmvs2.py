#!/usr/bin/env python3
#! Code for tuning the performance of pmvs2

#!/usr/bin/env python3

import numpy
import functools
import pathlib
from pathlib import Path

import networkx

from black_box_optimization import sequence, cartesian_product, greedy_neighbor_descent, exhaustive_search, graph_node_to_dict

from reconstruct_pmvs2 import PMVS2Options, run_pmvs

from benchmark import compare_clouds_and_load_runtime, stats_to_objective

# Define our search space
grid=cartesian_product((sequence([12],'numCameras'),
        sequence([0,1,2],'level'),
        sequence([2,4,8],'csize'),
        sequence([0.3, 0.5, 0.6, 0.7, 0.8],'threshold'),
        sequence([7],'wsize'),
        sequence([2,3],'minImageNum'),
        sequence([1,2,3,4,5,6,7,8,9,],'CPU'),
        sequence([1],'useVisData'),
        sequence([-1],'sequence'),
        sequence([None],'timages'),
        sequence([0],'oimages'),
        sequence([1,2,3],'numNeighbors')
        ))


# Set up our objective function
@functools.lru_cache(maxsize=None)
def f(parametersTuple):
    assert type(parametersTuple) is tuple
    d = graph_node_to_dict(graph=grid, node=parametersTuple)
    options = PMVS2Options(**d)

    widget = '2016_10_24__17_43_02'
    imagesPath = Path('data/undistorted_images') / widget
    workDirectory = Path('working_directory_pmvs')
    plyPath = Path('tuning_reconstruction.ply')
    runtimeFile = Path('tuning_runtime.txt')

    if runtimeFile.is_file():
        runtimeFile.unlink()
    if plyPath.is_file():
        plyPath.unlink()
    print('Performing the reconstruction...')
    run_pmvs(imagesPath,
             options=options,
             workDirectory=workDirectory,
             destFile=plyPath,
             runtimeFile=runtimeFile)

    print('Computing the objective')
    referencePath = Path('data') / 'reference_reconstructions' / widget / 'reference.ply'
    stats = compare_clouds_and_load_runtime(plyPath=plyPath,
                                            referencePath=referencePath,
                                            runtimeFile=runtimeFile)

    # Compute a scalar performance metric...
    objective = stats_to_objective(stats)
    return objective


seed = (12,1,2,0.7,7,2,4,1,-1,None,0,2) # TODO write converter from the class to dict.

stopPoint = greedy_neighbor_descent(grid, f, seed)
