#!/usr/bin/env python3

# Code for running the benchmark

from pathlib import Path
import numpy
import pandas
import datetime

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

raw = pandas.DataFrame()

for key in optionNames:
    print('Runing benchmark for algorithm key', key)
    for path in Path('./data/reconstructions').iterdir():

        assert path.is_dir(), "Unsure what to do with files in the reconstructions directory! There should only be folders here!"
        plyFiles = path.glob("*.ply")
        plyPath = path / (key + '.ply')

        assert plyPath in plyFiles, "Expected file " + str(plyPath) + ' does not exist! Try running reconstruct.py to add missing reconstructions?'
        cloud = load_ply(plyPath)[0][:, :3].astype(numpy.float32)

        scanID = path.name
        referencePath = Path('data/reference_reconstructions')/ scanID / 'reference.ply'
        assert referencePath.is_file(), 'Reference cloud file ' + str(referencePath) + ' could not be found... you probably need to make one manually in meshlab based on a high quality reconstruction?'
        referenceCloud = load_ply(referencePath)[0][:, :3].astype(numpy.float32)
        stats = compare_clouds(referenceCloud, cloud)
        stats['algorithm']=key
        stats['scanID'] = scanID
        #stats['t'] = to_datetime(scanID)
        #stats = stats.set_index('t')

        runtimeFile = path / (key+'_runtime.txt')
        with runtimeFile.open('r') as fd:
            dt = float(fd.readline()) # seconds
        stats['reconstructionTime'] = dt

        raw = raw.append(stats, ignore_index=False)

print('Done with all of the point cloud comparisons!')
#print(raw)

#raw.groupby('algorithm').mean()

# Extract more meaningful metrics from the raw comparison data. Smaller is better. 
metrics = pandas.DataFrame()
metrics['storageSize'] = raw['numCloud2Points'] / raw['numCloud1Points'].mean()
metrics['outlierRatio'] = (raw['numCloud2Points'] - raw['numCloud2PointsNearCloud1']) / raw['numCloud2Points']
metrics['incompletenessRatio'] = (raw['numCloud1Points'] - raw['numCloud1PointsNearCloud2']) / raw['numCloud1Points']
metrics['algorithm'] = raw['algorithm']
metrics['scanID'] = raw['scanID']
metrics['reconstructionTime'] = raw['reconstructionTime']
metrics.set_index('scanID')
metrics.to_csv('results/metrics.csv') # backup

# Ask some interesting questions about the data

# How did the algorithms do against each other on average?
byAlgorithm = metrics.groupby('algorithm').mean().sort_values('incompletenessRatio')
print(byAlgorithm)
with open('results/byAlgorithm.txt','w') as fd:
    fd.write(byAlgorithm.to_string()+'\n')
