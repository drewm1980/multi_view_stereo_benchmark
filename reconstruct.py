#!/usr/bin/env python3

# Code for updating all of the reconstructions

import pathlib
from pathlib import Path
import numpy

import sys

if __name__ == '__main__' and __package__ is None:
    sys.path.append(Path(__file__).absolute().parent)

allOptionNames = []
allOptionsDict = {}

# Enable PMVS reconstruction
import multi_view_stereo_benchmark # You probably need to add this to your PYTHONPATH!

from multi_view_stereo_benchmark.reconstruct_pmvs2 import run_pmvs, pmvsOptionNames, pmvsOptionsDict
allOptionNames += pmvsOptionNames
allOptionsDict.update(pmvsOptionsDict)

# Enable OpenCV reconstruction
from multi_view_stereo_benchmark.reconstruct_opencv import run_opencv, opencvOptionNames, opencvOptionsDict
allOptionNames += opencvOptionNames
allOptionsDict.update(opencvOptionsDict)

destFileNames = {optionName:optionName+'.ply' for optionName in allOptionNames}

def generate_missing_reconstructions(imagesPath: Path,
                                     reconstructions_path: Path=Path('.'),
                                     optionNames=allOptionNames,
                                     optionsDict=allOptionsDict,
                                     destFileNames=destFileNames):
    for optionName in allOptionNames:
        destFileName = destFileNames[optionName]
        if not (reconstructions_path / destFileName).is_file():
            print('Running reconstruction configuration: ', optionName)
            if 'pmvs' in optionName:
                run = run_pmvs
            elif 'opencv' in optionName:
                run = run_opencv
            else:
                assert False, "Couldn't figure out which algorithm to run from the name of the options!"
            run(imagesPath=imagesPath,
                     destDir=reconstructions_path,
                     options=optionsDict[optionName],
                     destFile=destFileNames[optionName])

def do_reconstructions_for_the_benchmark(sourceDir=Path('data/undistorted_images'),
                                         destDir=Path('data/reconstructions')):
    assert sourceDir.is_dir()
    if not destDir.is_dir():
        destDir.mkdir()
    for objectPath in sourceDir.iterdir():
        objectid = objectPath.name
        print('Performing reconstructions for object ', objectid)
        if not destDir.is_dir():
            destDir.mkdir()
        destPath = destDir / objectid
        if not destPath.is_dir():
            destPath.mkdir()
        generate_missing_reconstructions(imagesPath=objectPath,reconstructions_path=destPath)
        

if __name__=='__main__':
    do_reconstructions_for_the_benchmark()
    #generate_missing_reconstructions(imagesPath=Path('undistorted')) # All of them
    #generate_missing_reconstructions(imagesPath=Path('undistorted'), optionNames=['low']) # Just one
