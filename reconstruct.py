#!/usr/bin/env python3

from pathlib import Path
import numpy

from reconstruct_pmvs2 import set_up_visualize_subdirectory, run_pmvs

def generate_missing_reconstructions(imagesPath: Path,
                                     reconstructionsPath: Path=Path('.'),
                                     optionNames=optionNames,
                                     optionsDict=optionsDict,
                                     destFileNames=destFileNames):
    for optionName in optionNames:
        destFileName = destFileNames[optionName]
        if not (reconstructionsPath / destFileName).is_file():
            print('Running reconstruction configuration: ', optionName)
            run_pmvs(imagesPath=imagesPath,
                     destDir=reconstructionsPath,
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
        generate_missing_reconstructions(imagesPath=objectPath,reconstructionsPath=destPath)
        

if __name__=='__main__':
    do_reconstructions_for_the_benchmark()
    #generate_missing_reconstructions(imagesPath=Path('undistorted')) # All of them
    #generate_missing_reconstructions(imagesPath=Path('undistorted'), optionNames=['low']) # Just one
