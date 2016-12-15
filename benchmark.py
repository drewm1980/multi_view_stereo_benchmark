#!/usr/bin/env python3

# Code for running the benchmark

import pathlib
from pathlib import Path
import numpy
import datetime
import pandas

from compare_clouds import compare_clouds
from load_ply import load_ply
from reconstruct import optionNames

def to_datetime(scanID):
    ''' Convert my string timestamps to numpy timestamps '''
    year, month, day, _, hour, minute, second = scanID.split('_')
    strings = (year, month, day, hour, minute, second)
    ints = map(int, strings)
    ts = datetime.datetime(*ints)
    ts_numpy = numpy.datetime64(ts)
    return ts_numpy

# Note: These affect the optimization objective defined below!
typicalCloudSize = 32423 # Average number of points per reference cloud
typicalReconstructionTime = 1.0 # seconds.


def compare_clouds_and_load_runtime(plyPath, referencePath, runtimeFile):
    ''' Compute cloud similarity statistics given the Paths of two ply files
        and the path of a .txt file containing the runtime.'''
    cloud = load_ply(plyPath)[0][:, :3].astype(numpy.float32)
    referenceCloud = load_ply(referencePath)[0][:, :3].astype(numpy.float32)
    stats = compare_clouds(referenceCloud, cloud)
    with runtimeFile.open('r') as fd:
        dt = float(fd.readline()) # seconds
    stats['reconstructionTime'] = dt
    return stats


def stats_to_objective(stats):
    ''' This is where the master performance metric for the benchmark.
        This is intended to be used by parameter tuning code,
        and for sorting the final results of the benchmark. 
        lower objective is better.'''
    storageSize = stats['numCloud2Points'] / typicalCloudSize
    outlierRatio = (
        stats['numCloud2Points'] -
        stats['numCloud2PointsNearCloud1']) / stats['numCloud2Points']
    incompletenessRatio = (
        stats['numCloud1Points'] -
        stats['numCloud1PointsNearCloud2']) / stats['numCloud1Points']
    reconstructionTime = stats['reconstructionTime'] / typicalReconstructionTime
    meshQuality = (storageSize + outlierRatio + incompletenessRatio) / 3
    return (meshQuality + reconstructionTime) / 2


if __name__=='__main__':

    # Do all of our cloud comparisons, aggregating the data in a list of dicts
    rawStats = []
    for key in optionNames:
        print('Runing benchmark for algorithm key', key)
        for path in Path('./data/reconstructions').iterdir():

            assert path.is_dir(), "Unsure what to do with files in the reconstructions directory! There should only be folders here!"
            plyFiles = path.glob("*.ply")
            plyPath = path / (key + '.ply')

            assert plyPath in plyFiles, "Expected file " + str(plyPath) + ' does not exist! Try running reconstruct.py to add missing reconstructions?'

            scanID = path.name
            referencePath = Path('data/reference_reconstructions')/ scanID / 'reference.ply'
            assert referencePath.is_file(), 'Reference cloud file ' + str(referencePath) + ' could not be found... you probably need to make one manually in meshlab based on a high quality reconstruction?'

            runtimeFile = path / (key+'_runtime.txt')

            stats = compare_clouds_and_load_runtime(
                plyPath=plyPath, referencePath=referencePath, runtimeFile=runtimeFile)
            stats['algorithm']=key
            stats['scanID'] = scanID

            rawStats.append(stats)

    print('Done with all of the point cloud comparisons!')

    # Transpose to list of numpy arrays
    keys = list(rawStats[0].keys())
    ndarrays = []
    for key in keys:
        ndarray = numpy.array([stat[key] for stat in rawStats])
        ndarrays.append(ndarray)

    # Convert to pandas
    raw = pandas.DataFrame()
    columnsPandas = []
    for i in range(len(keys)):
        column = pandas.DataFrame(ndarrays[i],columns=[keys[i]])
        columnsPandas.append(column)
    raw = pandas.concat(columnsPandas, axis=1)

    #raw.groupby('algorithm').mean()

    # Extract more meaningful metrics from the raw comparison data. Smaller is better. 
    metrics = pandas.DataFrame()

    metrics['storageSize'] = raw['numCloud2Points'] / typicalCloudSize
    metrics['outlierRatio'] = (raw['numCloud2Points'] - raw['numCloud2PointsNearCloud1']) / raw['numCloud2Points']
    metrics['incompletenessRatio'] = (raw['numCloud1Points'] - raw['numCloud1PointsNearCloud2']) / raw['numCloud1Points']

    metrics['reconstructionTime'] = raw['reconstructionTime']
    metrics['algorithm'] = raw['algorithm']
    metrics['scanID'] = raw['scanID']
    metrics.set_index('scanID')
    metrics.to_csv('results/metrics.csv') # backup

    # Ask some interesting questions about the data

    # How did the algorithms do against each other on average?
    byAlgorithm = metrics.groupby('algorithm').mean().sort_values('incompletenessRatio')
    print(byAlgorithm)
    with open('results/byAlgorithm.txt','w') as fd:
        fd.write(byAlgorithm.to_string()+'\n')
