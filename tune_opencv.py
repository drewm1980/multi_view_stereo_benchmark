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

def nested_dict_to_list_of_tuples(d, separator='.', extra_nesting=False):
    """ Take a 2-level nested dict of dict to list of tuples.
        to preserve some structure, the keys from the two levels are 
        joined with the separator string.
        If extra_nesting is true, the inner items are wrapped in an extra
        tuple. 
        This function is mainly to support use with my black box tuning code,
        which represents the search space as key-tuple(value) pairs."""
    l = []
    for outer_key, outer_value in d.items():
        assert separator not in outer_key, 'Please try another separator!'
        for inner_key, inner_value in outer_value.items():
            assert separator not in inner_key, 'Please try another separator!'
            if extra_nesting:
                inner_value = [inner_value]
            l.append((outer_key + separator + inner_key, inner_value))
    return l
#nested_dict_to_list_of_tuples(DefaultOptionsBM,extra_testing=True)


def unmangle_tuples_to_nested_dict(l, separator='.'):
    """ Inverse of the nested_dict_to_list_of_tuples function, assuming there
        were no name collisions in the name mangling, and no extra_nesting. """
    d = {}
    for outer_inner_key, inner_value in l:
        outer_key, inner_key = outer_inner_key.split(separator)
        if outer_key not in d:
            d[outer_key] = {}
        assert inner_key not in d[outer_key], 'Name collision!'
        d[outer_key][inner_key] = inner_value
    return d


# An array of image sizes that preserve the image aspect ratio
native_resolution = numpy.array((1280,1024))
ratios = numpy.linspace(0.25, 1.0, 10)
resolutions = [tuple(numpy.floor_divide(native_resolution,1/ratio).astype('int')) for ratio in ratios]

grid = cartesian_product((
    sequence('CameraArray.channels', [1]),
    #sequence('CameraArray.topology', ['overlapping','adjacent','skipping_1','skipping_2']),
    #sequence('CameraArray.topology', ['adjacent', 'skipping_1']),
    sequence('CameraArray.topology', ['skipping_1']),
    sequence('CameraArray.num_cameras', [12]),
    sequence('Remap.interpolation', [1]),
    #sequence('StereoMatcher.MinDisparity', [16*n for n in range(-5,20)]),
    sequence('StereoMatcher.MinDisparity', [0]),
    #sequence('StereoMatcher.NumDisparities', [10 * 16, 20 * 16, 30 * 16]),
    #sequence('StereoMatcher.NumDisparities', [16*n for n in range(10,30)]),
    sequence('StereoMatcher.NumDisparities', [288]),
    sequence('StereoMatcher.SpeckleWindowSize', [0]),
    sequence('StereoMatcher.SpeckleRange', [0]),
    #sequence('StereoMatcher.Disp12MaxDiff', [0,1,2,3,4,5]),
    sequence('StereoMatcher.Disp12MaxDiff', [0]),
    #sequence('StereoMatcher.BlockSize', [9, 11, 13, 15, 17, 19, 21, 23, 25]),
    sequence('StereoMatcher.BlockSize', [21]),
    #sequence('StereoRectify.newImageSize', resolutions),
    #sequence('StereoRectify.newImageSize', [(640, 512)]),
    sequence('StereoRectify.newImageSize', [(1280, 1024)]),
    sequence('StereoRectify.imageSize', [(1280, 1024)]),
    sequence('StereoRectify.flags', [0]),
    #sequence('StereoRectify.alpha', [0.0, 0.25, 0.5, 0, 0.75, 1.0]),
    #sequence('StereoRectify.alpha', [0.0, 0.5, 1.0]),
    sequence('StereoRectify.alpha', [0.5]),
    sequence('StereoBM.PreFilterCap', [i for i in range(5,64,5)]+[63]),
    #sequence('StereoBM.PreFilterCap', [63]),
    #sequence('StereoBM.PreFilterSize', [5,7,9,13]),
    sequence('StereoBM.PreFilterSize', [5]),
    #sequence('StereoBM.UniquenessRatio', [10]),
    sequence('StereoBM.UniquenessRatio', [i for i in range(5,11)]),
    #complete('StereoBM.PreFilterType',
             #[cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE,
              #cv2.StereoBM_PREFILTER_XSOBEL]),
    sequence('StereoBM.PreFilterType',[0]),
    #sequence('StereoBM.TextureThreshold', [10]),
    sequence('StereoBM.TextureThreshold', [n for n in range(20)]),
    #sequence('StereoBM.SmallerBlockSize', [80]), # Dead code in opencv!
    ))


seed_tuple_of_tuple=(('CameraArray.channels', 1),
 ('CameraArray.topology', 'skipping_1'),
 ('CameraArray.num_cameras', 12),
 ('Remap.interpolation', 1),
 ('StereoMatcher.MinDisparity', 0),
 ('StereoMatcher.NumDisparities', 288),
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
 )

seed_tuple = tuple((value for key, value in seed_tuple_of_tuple))

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
temp_matcher = OpenCVStereoMatcher(calibration_path=imagesPath)
temp_matcher.load_images(imagesPath)
images = temp_matcher.images
del temp_matcher

# Set up pretty printing of options
try:
    from IPython.lib.pretty import pprint
    print('The pretty printer from ipython is available!')
except:
    pprint = print
def pretty_print_options(options):
    l = nested_dict_to_list_of_tuples(options)
    pprint(l)

def pretty_print(point_tuple):
    """ Pretty print the tuple representation of the options. """
    pprint(tuple(zip(parameterNames,point_tuple)))

def print_options_as_nested_dict(point_tuple):
    pprint(unmangle_tuples_to_nested_dict(tuple(zip(parameterNames,point_tuple))))

# Set up our objective function
@functools.lru_cache(maxsize=None)
def f(parametersTuple):
    assert type(parametersTuple) is tuple
    options = unmangle_tuples_to_nested_dict(zip(parameterNames,parametersTuple))
    #pretty_print_options(options)

    matcher = OpenCVStereoMatcher(options=options, calibration_path=imagesPath)

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

seed_tuple = greedy_neighbor_descent(grid, f, seed=seed_tuple, pretty_print=pretty_print)
#print(seed,' -> ',f(seed))
f.cache_clear()
f(seed_tuple)
print('Result as list of tuples for pasting as a saved seed in tune_opencv.py:')
pretty_print(seed_tuple)
print('Result as nested dicts for pasting into reconstruct_opencv.py:')
print_options_as_nested_dict(seed_tuple)



#tunedOptions = PMVS2Options(**dict(zip(parameterNames,seed)))
#print('PMVS2Options(',','.join(tuple(map('='.join,zip(parameterNames,map(str,seed))))),')')
