#!/usr/bin/env python3
#! Code for tuning the performance of the opencv stereo matcher

import numpy
import functools
import pathlib
from pathlib import Path

import networkx

from black_box_optimization import sequence, cartesian_product, strong_product, greedy_neighbor_descent, exhaustive_search, complete

from reconstruct_opencv import OpenCVStereoMatcher, StereoBMOptions

from benchmark import stats_to_objective
from compare_clouds import compare_clouds
from load_ply import load_ply, save_ply

import cv2

# An array of image sizes that preserve the image aspect ratio
native_resolution = numpy.array((1280,1024))
ratios = numpy.linspace(0.25, 1.0, 10)
resolutions = [tuple(numpy.floor_divide(native_resolution,1/ratio).astype('int')) for ratio in ratios]

grid = cartesian_product((
    sequence('CameraArray.channels', [1]),
    #sequence('CameraArray.topology', ['overlapping','adjacent','skipping_1','skipping_2']),
    sequence('CameraArray.topology', ['adjacent', 'skipping_1']),
    sequence('CameraArray.num_cameras', [12]),
    sequence('Remap.interpolation', [1]),
    sequence('StereoMatcher.MinDisparity', [10]),
    sequence('StereoMatcher.NumDisparities', [10 * 16, 20 * 16, 30 * 16]),
    sequence('StereoMatcher.SpeckleWindowSize', [0]),
    sequence('StereoMatcher.SpeckleRange', [0]),
    sequence('StereoMatcher.Disp12MaxDiff', [0]),
    sequence('StereoMatcher.BlockSize', [9, 11, 13, 15, 17, 19, 21, 23, 25]),
    sequence('StereoRectify.newImageSize', resolutions),
    sequence('StereoRectify.imageSize', [(1280, 1024)]),
    sequence('StereoRectify.flags', [0]),
    sequence('StereoRectify.alpha', [0.0, 0.25, 0.5, 0, 0.75, 1.0]),
    sequence('StereoBM.PreFilterCap', [63]),
    sequence('StereoBM.PreFilterSize', [5]),
    sequence('StereoBM.UniquenessRatio', [10]),
    complete('StereoBM.PreFilterType',
             [cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE,
              cv2.StereoBM_PREFILTER_XSOBEL]),
    sequence('StereoBM.TextureThreshold', [10]),
    sequence('StereoBM.SmallerBlockSize', [80])))

seed = (('CameraArray.channels', 1),
        ('CameraArray.topology', 'adjacent'),
        ('CameraArray.num_cameras', 12),
        ('Remap.interpolation', 1),
        ('StereoMatcher.MinDisparity', 10),
        ('StereoMatcher.NumDisparities', 320),
        ('StereoMatcher.SpeckleWindowSize', 0),
        ('StereoMatcher.SpeckleRange', 0),
        ('StereoMatcher.Disp12MaxDiff', 0),
        ('StereoMatcher.BlockSize', 21),
        ('StereoRectify.newImageSize', (1280, 1024)),
        ('StereoRectify.imageSize', (1280, 1024)),
        ('StereoRectify.flags', 0),
        ('StereoRectify.alpha', 0.5),
        ('StereoBM.PreFilterCap', 63),
        ('StereoBM.PreFilterSize', 5),
        ('StereoBM.UniquenessRatio', 10),
        ('StereoBM.PreFilterType', 0),
        ('StereoBM.TextureThreshold', 10),
        ('StereoBM.SmallerBlockSize', 80), )
seed = tuple((value for key, value in seed))

parameterNames = grid.name.split(',')

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

from reconstruct_opencv import unmangle_tuples_to_nested_dict


# Set up our objective function
@functools.lru_cache(maxsize=None)
def f(parametersTuple):
    assert type(parametersTuple) is tuple
    options = unmangle_tuples_to_nested_dict(zip(parameterNames,parametersTuple))
    try:
        from IPython.lib import pretty
        pretty(options)
    except:
        print(options)

    matcher = OpenCVStereoMatcher(options=options, calibrationsPath=imagesPath)

    try:
        xyz, reconstruction_time = matcher.run_from_memory(images)
        print('Saving .ply file to',plyPath,'...')
        save_ply(xyz=xyz,filename=plyPath)
        with open(str(runtimeFile), 'w') as fd:
            fd.write(str(reconstruction_time)) # seconds
    except Exception as e:
        print('OpenCVStereoMatcher.run_from_memory threw an exception!')
        print(str(e))
        print('Returning infinite objective!')
        return float('inf')

    print('Computing the objective...')
    try:
        stats = compare_clouds(referenceCloud, xyz)
        stats['reconstructionTime'] = reconstruction_time
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

seed = greedy_neighbor_descent(grid, f, seed=seed)
print(seed,' -> ',f(seed))


#tunedOptions = PMVS2Options(**dict(zip(parameterNames,seed)))
#print('PMVS2Options(',','.join(tuple(map('='.join,zip(parameterNames,map(str,seed))))),')')
