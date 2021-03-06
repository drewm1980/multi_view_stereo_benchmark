#!/usr/bin/env python3
#! Code for tuning the performance of pmvs2

import numpy
import functools
import pathlib
from pathlib import Path

import networkx

from black_box_optimization import sequence, cartesian_product, strong_product, greedy_neighbor_descent, exhaustive_search

from reconstruct_pmvs2 import PMVS2Options, run_pmvs

from benchmark import compare_clouds_and_load_runtime, stats_to_objective

# Define our search space
grid=cartesian_product((sequence('numCameras',[12]),
        sequence('level',[0,1,2]),
        sequence('csize',[2,4,8]),
        sequence('threshold',[0.3, 0.5, 0.6, 0.7, 0.8]),
        sequence('wsize',[7]),
        sequence('minImageNum',[2,3]),
        sequence('CPU',[1,2,3,4,5,6,7,8,9,]),
        sequence('useVisData',[1]),
        sequence('sequence',[-1]),
        sequence('timages',[None]),
        sequence('oimages',[0]),
        sequence('numNeighbors',[1,2,3]),
        ))

parameterNames = grid.name.split(',')


# Set up our objective function
@functools.lru_cache(maxsize=None)
def f(parametersTuple):
    assert type(parametersTuple) is tuple

    d = dict(zip(parameterNames,parametersTuple))
    print(d)
    options = PMVS2Options(**d)

    widget = '2016_10_24__17_43_02'
    imagesPath = Path('data/undistorted_images') / widget
    workDirectory = Path('working_directory_pmvs')
    plyPath = workDirectory / 'tuning_reconstruction.ply'
    runtimeFile = workDirectory / 'tuning_runtime.txt'

    if runtimeFile.is_file():
        runtimeFile.unlink()
    if plyPath.is_file():
        plyPath.unlink()
    print('Performing the reconstruction...')

    try:
        run_pmvs(imagesPath,
                 options=options,
                 workDirectory=workDirectory,
                 destFile=plyPath,
                 runtimeFile=runtimeFile)
    except:
        print('run_pmvs threw an exception!')
        print('Returning infinite objective!')
        return float('inf')

    if not plyPath.is_file():
        print("run_pmvs didn't generate ", plyPath,
              "Returning infinite objective!")
        return float('inf')

    if not runtimeFile.is_file():
        print('No runtime file was generated!')
        print('Returning infinite objective!')
        return float('inf')

    print('Computing the objective...')
    referencePath = Path('data') / 'reference_reconstructions' / widget / 'reference.ply'

    try:
        stats = compare_clouds_and_load_runtime(plyPath=plyPath,
                                                referencePath=referencePath,
                                                runtimeFile=runtimeFile)
    except:
        print('There was an exception in compare_clouds_and_load_runtime!')
        print('Returning infinite objective!')
        return float('inf')

    # Compute a scalar performance metric...
    objective = stats_to_objective(stats)
    return objective

# Tuning Methodology:
# Run this script in ipython to get the above functions
# and memoized objective function defined above.
# Then copy and paste stages of optimization into your ipython terminal.
# Optimization steps that actually make progress can be put here for posterity,
# or manually added to the benchmark.

seed = (12, 2, 8, 0.8, 7, 3, 7, 1, -1, None, 0, 2)

grid=strong_product((sequence('numCameras',[12]),
        sequence('level',[0,1,2]),
        sequence('csize',[2,4,8]),
        sequence('threshold',[0.3, 0.5, 0.6, 0.7, 0.8]),
        sequence('wsize',[7]),
        sequence('minImageNum',[2,3]),
        sequence('CPU',[1,2,3,4,5,6,7,8,9,]),
        sequence('useVisData',[1]),
        sequence('sequence',[-1]),
        sequence('timages',[None]),
        sequence('oimages',[0]),
        sequence('numNeighbors',[1,2,3]),
        ))
seed = greedy_neighbor_descent(grid, f, seed=seed)
#(12, 2, 8, 0.8, 7, 3, 8, 1, -1, None, 0, 3) -> 0.52

grid=strong_product((sequence('numCameras',[12]),
        sequence('level',[2,3]),
        sequence('csize',[4,5,6,7,8,10,12]),
        sequence('threshold',[0.4, 0.5, 0.55, 0.6, 0.65 ]),
        sequence('wsize',[6,7,8,9,10,11]),
        sequence('minImageNum',[2,3,4]),
        sequence('CPU',[7,8]),
        sequence('useVisData',[1]),
        sequence('sequence',[-1]),
        sequence('timages',[None]),
        sequence('oimages',[0]),
        sequence('numNeighbors',[2,3]),
        ))
seed = greedy_neighbor_descent(grid, f, seed=seed)
#(12, 3, 5, 0.6, 6, 3, 7, 1, -1, None, 0, 2)  ->  0.420616645319
#(12, 3, 5, 0.6, 6, 3, 7, 1, -1, None, 0, 2)  ->  0.420616645319
print(seed,' -> ',f(seed))

#tunedOptions = PMVS2Options(**dict(zip(parameterNames,seed)))

print('PMVS2Options(',','.join(tuple(map('='.join,zip(parameterNames,map(str,seed))))),')')
