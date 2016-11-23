#!/usr/bin/env python3
#! Code for tuning the performance of pmvs2

#!/usr/bin/env python3

import numpy
import functools
import pathlib
from pathlib import Path

import networkx

from black_box_optimization import sequence, cartesian_product, strong_product, greedy_neighbor_descent, exhaustive_search

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


seed = (12,1,2,0.7,7,2,4,1,-1,None,0,2) # -> 42

# To view progress taking advantage of the above cache,
# First run this script in ipython, then re-run this line:
stopPoint = greedy_neighbor_descent(grid, f, seed)
#(12, 2, 8, 0.8, 7, 3, 7, 1, -1, None, 0, 2) -> 0.986

grid2=strong_product((sequence([12],'numCameras'),
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
stopPoint2 = greedy_neighbor_descent(grid2, f, seed=stopPoint)
#(12, 2, 8, 0.8, 7, 3, 7, 1, -1, None, 0, 2) -> 0.968


grid3=strong_product((sequence([12],'numCameras'),
        sequence([1,2,3],'level'),
        sequence([4,8,16],'csize'),
        sequence([0.6, 0.7, 0.8, 0.9, 1.0],'threshold'),
        sequence([7],'wsize'),
        sequence([2,3],'minImageNum'),
        sequence([4,5,6,7,8,9,],'CPU'),
        sequence([1],'useVisData'),
        sequence([-1],'sequence'),
        sequence([None],'timages'),
        sequence([0],'oimages'),
        sequence([1,2,3],'numNeighbors')
        ))
stopPoint3 = greedy_neighbor_descent(grid3, f, seed=stopPoint2)
#(12, 3, 8, 0.8, 7, 3, 8, 1, -1, None, 0, 2) -> 0.82

grid4=strong_product((sequence([12],'numCameras'),
        sequence([2,3],'level'),
        sequence([4,6,8,10,12],'csize'),
        sequence([0.7, 0.75, 0.8, 0.85, 0.9],'threshold'),
        sequence([7],'wsize'),
        sequence([2,3],'minImageNum'),
        sequence([7,8],'CPU'),
        sequence([1],'useVisData'),
        sequence([-1],'sequence'),
        sequence([None],'timages'),
        sequence([0],'oimages'),
        sequence([1,2,3],'numNeighbors')
        ))
stopPoint4 = greedy_neighbor_descent(grid4, f, seed=stopPoint3)
#(12, 3, 6, 0.7, 7, 3, 8, 1, -1, None, 0, 2) -> 0.82

grid5=strong_product((sequence([12],'numCameras'),
        sequence([2,3,4],'level'),
        sequence([4,5,6,7,8,10,12],'csize'),
        sequence([0.6, 0.7, 0.75, 0.8, 0.85, 0.9],'threshold'),
        sequence([6,7,8],'wsize'),
        sequence([2,3,4],'minImageNum'),
        sequence([7,8],'CPU'),
        sequence([1],'useVisData'),
        sequence([-1],'sequence'),
        sequence([None],'timages'),
        sequence([0],'oimages'),
        sequence([2],'numNeighbors')
        ))
stopPoint5 = greedy_neighbor_descent(grid5, f, seed=stopPoint4)
#(12, 3, 6, 0.7, 8, 3, 7, 1, -1, None, 0, 2) -> 0.8175

grid6=cartesian_product((sequence([12],'numCameras'),
        sequence([3],'level'),
        sequence([4,5,6,7,8,10,12],'csize'),
        sequence([0.6, 0.65, 0.7, 0.73, 0.75, 0.8, 0.85, 0.9],'threshold'),
        sequence([6,7,8,9,10,11],'wsize'),
        sequence([3],'minImageNum'),
        sequence([7],'CPU'),
        sequence([1],'useVisData'),
        sequence([-1],'sequence'),
        sequence([None],'timages'),
        sequence([0],'oimages'),
        sequence([2],'numNeighbors')
        ))
stopPoint6 = greedy_neighbor_descent(grid6, f, seed=stopPoint5)
#(12, 3, 6, 0.7, 8, 3, 7, 1, -1, None, 0, 2) -> 0.8175 (Same point)

tunedOptions = PMVS2Options(**dict(zip(parameterNames,stopPoint6)))

