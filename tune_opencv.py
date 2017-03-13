#!/usr/bin/env python3
#! Code for tuning the performance of the opencv stereo matcher

import numpy
import functools
import pathlib
from pathlib import Path

import networkx

from black_box_optimization import sequence, cartesian_product, strong_product, greedy_neighbor_descent, exhaustive_search

from reconstruct_opencv import OpenCVStereoMatcher, StereoBMOptions

from benchmark import stats_to_objective
from compare_clouds import compare_clouds
from load_ply import load_ply, save_ply

# Define our search space. Search within the normal block matcher to start with...
grid=cartesian_product(( sequence([0.0,0.25,0.5,0.75,1.0],'alpha'),
        sequence([(0,0)],'newImageSize'),
        sequence([10*16,20*16,30*16],'numDisparities'),
        sequence([9,15,19,21,23,25],'blockSize'),
        sequence(['overlapping','adjacent','skipping_1','skipping_2'],'topology')
        ))

parameterNames = grid.name.split(',')

seed = (0.5, (0,0), 20*16, 21,'overlapping') # Known to work


# Locations of data used in the benchmark
widget = '2016_10_24__17_43_02'
imagesPath = Path('data/undistorted_images') / widget
workDirectory = Path('working_directory_opencv')
plyPath = workDirectory / 'tuning_reconstruction.ply'
runtimeFile = workDirectory / 'tuning_runtime.txt'

# Load the reference point cloud
referencePath = Path('data') / 'reference_reconstructions' / widget / 'reference.ply'
referenceCloud = load_ply(referencePath)[0][:, :3].astype(numpy.float32)

# Load the images once.
temp_matcher = OpenCVStereoMatcher(calibrationsPath=imagesPath)
temp_matcher.load_images(imagesPath)
images = temp_matcher.images
del temp_matcher


# Set up our objective function
@functools.lru_cache(maxsize=None)
def f(parametersTuple):
    assert type(parametersTuple) is tuple
    d = dict(zip(parameterNames,parametersTuple))
    print(d)

    d2 = d.copy()
    del d2['topology']
    matcher_options = StereoBMOptions(**d2)
    matcher = OpenCVStereoMatcher(topology=d['topology'],
                                  calibrationsPath=imagesPath,
                                  matcher_options=matcher_options)

    try:
        xyz, dt = matcher.run_from_memory(images)
        save_ply(xyz=xyz,filename=plyPath)
    except Exception as e:
        print('OpenCVStereoMatcher.run_from_memory threw an exception!')
        print(str(e))
        print('Returning infinite objective!')
        return float('inf')


    print('Computing the objective...')
    try:
        stats = compare_clouds(referenceCloud, xyz)
        stats['reconstructionTime'] = dt
    except Exception as e:
        print('There was an exception in compare_clouds!')
        print(str(e))
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

print(seed,' -> ',f(seed))
seed = greedy_neighbor_descent(grid, f, seed=seed)


#tunedOptions = PMVS2Options(**dict(zip(parameterNames,seed)))
#print('PMVS2Options(',','.join(tuple(map('='.join,zip(parameterNames,map(str,seed))))),')')
