#!/usr/bin/env python3

# Code for actually running the benchmark

from pathlib import Path
from compare_clouds import compare_clouds
from load_ply import load_ply
import numpy

referenceKey = 'reference'

keysToBenchmark = ('high_quality', 'medium_quality', 'low_quality')

for key in keysToBenchmark:
    for path in Path('./data/reconstructions').iterdir():

        assert path.is_dir(), "Unsure what to do with files in the reconstructions directory! There should only be folders here!"
        plyFiles = path.glob("*.ply")
        plyPath = path / (key + '.ply')

        assert plyPath in plyFiles, "Expected file " + str(plyPath) + ' does not exist! Try running reconstruct.py to add missing reconstructions?'
        cloud = load_ply(plyPath)[0][:, :3].astype(numpy.float32)

        referencePath = path / (referenceKey + '.ply')
        assert referencePath.is_file(), 'Reference cloud file ' + str(referencePath) + ' could not be found... you probably need to make one manually in meshlab based on a high quality reconstruction?'
        referenceCloud = load_ply(referencePath)[0][:, :3].astype(numpy.float32)

        compare_clouds(referenceCloud, cloud)
