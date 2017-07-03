#!/usr/bin/env python3

# Code for performing reconstruction using pmvs2

import numpy
import pathlib
from pathlib import Path

import os
import inspect
pwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
pmvs2Path = Path(pwd) / 'extern/CMVS-PMVS/program/main/pmvs2'
assert pmvs2Path.is_file(), "pmvs2 binary not found. Try running bootstrap.sh?"

from .load_camera_info import load_intrinsics, load_extrinsics


def set_up_visualize_subdirectory(images=None, inputPath=None, destPath=None):
    """
    Create the "visualize" subdirectory required by PMVS
    Inputs:
    inputPath -- full path to a directory containing undistorted images
    destPath -- full path to a directory where the "visualize" subdir will be 
                created
    """
    assert destPath is not None: "destPath is a required argument!"
    assert (images is None)!=(inputPath is None), 'Please pass either "images" or "inputPath" but not both!'

    visualizePath = destPath / 'visualize'
    if not visualizePath.is_dir():
        visualizePath.mkdir()

    print('Setting up visualize subdirectory in ' + str(visualizePath)+'...')

    import glob
    numCameras = len(list((inputPath.glob("*.png"))))
    for i in range(numCameras):
        sourceFilename = "image_camera%02i.png"%(i+1)
        destFilename = "%08i.ppm"%(i)
        sourcePath = inputPath / sourceFilename
        destPath = visualizePath / destFilename
        # Call image magick as a binary to convert the images
        args = ['convert',str(sourcePath),str(destPath)]
        print('Running command: ' + ' '.join(args)+' ...')
        import subprocess
        subprocess.check_output(args=args)


def set_up_txt_subdirectory(inputPath,destPath):
    """ Generate the txt/*.txt files that PMVS uses to
        input the projection matrices for the images it runs on.
        Input:
        inputPath -- The directory containing the undistorted images,
                    and HALCON text files containing the intrinsics
                    and extrinsics of the images.
                    The names of the .txt files must be based on
                    names of the image files.
        destPath -- full path to a directory where the "txt" subdir will be 
                    created
    """

    numCameras = len(list(inputPath.glob('*.png')))

    txtPath = destPath / 'txt'
    if not txtPath.is_dir():
        txtPath.mkdir()

    for i in range(numCameras):
        # Load the intrinsics
        intrinsicsFilePath = inputPath / ('intrinsics_camera%02i.txt' % (i + 1))
        assert intrinsicsFilePath.is_file(), "Couldn't find camera intrinsics in "+str(intrinsicsFilePath)
        cameraMatrix, distCoffs, imageWidth, imageHeight = load_intrinsics(intrinsicsFilePath)
        ## The images must already be radially undistorted!
        #assert(abs(distCoffs[0]) < .000000001)
        #assert(abs(distCoffs[1]) < .000000001)
        #assert(abs(distCoffs[2]) < .000000001)
        #assert(abs(distCoffs[3]) < .000000001)
        #assert(abs(distCoffs[4]) < .000000001)

        # Load the extrinsics
        extrinsicsFilePath = inputPath / ('extrinsics_camera%02i.txt' % (i + 1))
        R, T = load_extrinsics(extrinsicsFilePath)

        # PMVS expects the inverse of the transform that HALCON exports!
        R,T = R.T,numpy.dot(-R.T,T)

        # Compute the projection matrix pmvs2 wants,
        # which combines intrinsics and extrinsics
        temp = numpy.hstack((R,numpy.reshape(T,(3,1)))) # 3x4

        P = numpy.dot(cameraMatrix,temp) # 3x4

        outFilePath = txtPath / ('%08i.txt'%i)
        numpy.savetxt(str(outFilePath),P,'%f',header='CONTOUR',comments='')


class PMVS2Options():
    """ This class represents most of the user-supplied options to PMVS2.
        It has sane default arguments that you can individually over-ride
        when you instantiate it.
        
        For argument descriptions see http://www.di.ens.fr/pmvs/documentation.html
        """
    def __init__(self,
                 numCameras,
                 level=1, # 0 is full resolution
                 csize=2, # cell size
                 threshold=0.6,
                 wsize=7, # colors
                 minImageNum=2,
                 #CPU=8, # For a quad core with hyperthreading
                 #CPU=20, # For a 20 core with hyperthreading
                 CPU=40, # For a 20 core with hyperthreading
                 #CPU=80, # For a 20 core with hyperthreading
                 #CPU=10, # For a 20 core with hyperthreading
                 #CPU=20, # For a 20 core with hyperthreading
                 #CPU=1, # For a 20 core with hyperthreading
                 useVisData=1,
                 sequence=-1,
                 timages=None,
                 oimages=0,
                 numNeighbors=2):
        self.level = level
        self.csize = csize
        self.threshold = threshold
        self.wsize = wsize
        self.minImageNum = minImageNum
        self.CPU = CPU
        self.useVisData = useVisData
        self.sequence = sequence
        self.numNeighbors = numNeighbors
        if timages is None:
            self.timages = (-1, 0, numCameras)
        else:
            self.timages = timages
        self.oimages = oimages

    def __hash__(self):
        fields = list(self.__dict__.items())
        fields.sort()
        return hash(tuple(fields)) # 

    def write_options_file(self,
                           optionsDir=Path('.'),
                           optionsFile=Path('option.txt')):
        optionsFilePath = optionsDir / optionsFile
        with optionsFilePath.open('w') as fd:
            for key,val in vars(self).items():
                if key == 'numNeighbors':
                    continue # numNeighbors doesn't go in the options file!
                if type(val) in (int,float):
                    fd.write('%s %s\n'%(key,str(val)))
                    continue
                if type(val) in (list,tuple):
                    fd.write(key + ' ' + ' '.join(map(str, val)) + '\n')
                    continue

def write_vis_file_ring(numCameras,numNeighbors=1,visFilePath=Path('vis.dat')):
    """ Generate a vis.dat file that pmvs2 expects for a camera array with 
    ring topology and a configurable number of neighbors to be used for reconstruction 
    
    Inputs:
    numCameras -- The number of cameras in the ring
    numNeighbors -- For any camera, the number of other adjacent cameras to use for matching
                    i.e. 1 for stereo, 2 for trinocular...
    """
    with visFilePath.open('w') as fd:
        fd.write('VISDATA\n')
        fd.write(str(numCameras)+'\n')
        assert(numNeighbors >= 1)
        assert(numNeighbors+1)
        for center_camera in range(numCameras):
            numPositiveNeighbors = int(numNeighbors)//2 + numNeighbors%2
            numNegativeNeighbors = int(numNeighbors)//2
            fd.write(str(center_camera)+' ')
            fd.write(str(numNeighbors)+' ')
            for i in range(numPositiveNeighbors):
                neighbor_camera = (center_camera+i+1)%numCameras
                fd.write(str(neighbor_camera) + ' ')
            for i in range(numNegativeNeighbors):
                neighbor_camera = (center_camera-i-1)%numCameras
                fd.write(str(neighbor_camera) + ' ')
            fd.write('\n')


def set_up_pmvs_tree(images=None, all_camera_parameters=None, inputPath=None, destPath=None, options=None):
    """ Set up a PMVS style file tree in destPath.
    inputPath contains images and HALCON camera parameter files."""
    assert destPath is not None, "set_up_pmvs_tree requires a destination path!"

    assert (images is None)==(all_camera_parameters is None), "Please pass both or neither of images and all_camera_parameters"
    assert (images is None)!=(inputPath is None), 'Please pass either "images" or "inputPath" but not both!'

    set_up_visualize_subdirectory(images=images,inputPath=inputPath,destPath=destPath)
    set_up_txt_subdirectory(inputPath=inputPath,destPath=destPath)

    # Generate the empty directory where pmvs puts its ply files
    modelsDir = destPath / 'models'
    if not modelsDir.is_dir():
        modelsDir.mkdir()

    numCameras = len(list(inputPath.glob('*.png')))

    # Generate PMVS options file
    if options is None:
        options = PMVS2Options(numCameras=numCameras)
    options.write_options_file(optionsDir=destPath,
                               optionsFile='option.txt')

    # Generate PMVS vis.dat file
    write_vis_file_ring(numCameras=numCameras,
                        numNeighbors=options.numNeighbors,
                        visFilePath=destPath / 'vis.dat')


def run_pmvs(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None):
    """ Run PMVS2 on a directory full of images.

        The images must ALREADY be radially undistorted!

    Arguments:
    imagesPath -- A directory full of source images
    destDir -- The destination directory of the ply file. (default current directory)
    destFile -- The destination name of the ply file. (default <name of the directory>.ply)
    options -- An instance of PMVS2Options
    workDirectory -- Existing directory where pmvs will work. (default generates a temp directory)
    runtimeFile -- The name of a file where info regarding the runtime will be stored.
    """
    import shutil
    import glob

    # By default, work in a temporary directory.
    # "with...as" ensures the temp directory is cleared even if there is an error below.
    if workDirectory is None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=str(Path(pwd)/'tmp')) as workDirectory:
            run_pmvs(imagesPath=imagesPath,
                     destDir=destDir,
                     destFile=destFile,
                     options=options,
                     runtimeFile=runtimeFile,
                     workDirectory=Path(workDirectory))
        return
    if not workDirectory.is_dir():
        workDirectory.mkdir()

    imagesPath = imagesPath.resolve()

    set_up_pmvs_tree(inputPath=imagesPath,
                     destPath=workDirectory,
                     options=options)

    # Run PMVS2
    import subprocess
    from time import time
    args = [str(pmvs2Path), './', str('option.txt')] # Careful! That damn slash after the dot is CRITICAL
    print('Running command ', ' '.join(args))
    t1 = time()
    #result = subprocess.run(args=args, cwd=str(workDirectory), stdout=subprocess.PIPE) # Python 3.5
    #stdout = result.stdout
    #returncode = result.returncode
    proc = subprocess.Popen(args=args, cwd=str(workDirectory), stdout=subprocess.PIPE) # Python 3.4
    stdout, stderr = proc.communicate()
    returncode = proc.returncode

    t2 = time()
    dt = t2-t1 # seconds. TODO: scrape more accurate timing from PMVS shell output
    print("pmvs2 output:")
    print(stdout.decode('utf8'))
    if returncode != 0:
        print("WARNING! pmvs2 returned a non-zero return value!")

    # Copy the file to the appropriate destination
    if destDir is None:
        destDir = Path.cwd()
    if destFile is None:
        destFile = 'reconstruction.ply'
    destPath = destDir / destFile

    if runtimeFile is None:
        runtimeFile = destPath.parent / (destPath.stem +'_runtime.txt')
    with open(str(runtimeFile), 'w') as fd:
        fd.write(str(dt)) # seconds

    modelsDir = workDirectory / 'models'
    plyPath = modelsDir / Path('option.txt' + '.ply')
    if plyPath.is_file():
        plyPath.rename(destPath)
    else:
        print(".ply file wasn't generated!")
        print('modelsDir: ' + str(modelsDir))
        print('plyPath: ' + str(plyPath))
        assert False


# Some hard-coded options, roughly slow to fast
pmvsOptionsDict = {
                                #'pmvs_tuned1': PMVS2Options(minImageNum=3,
                                #CPU=7,
                                #useVisData=1,
                                #numNeighbors=2,
                                #oimages=0,
                                #sequence=-1,
                                #wsize=8,
                                #numCameras=12,
                                #timages=None,
                                #level=3,
                                #threshold=0.7,
                                #csize=6),
                                #'pmvs_tuned2': PMVS2Options(numCameras=12,
                                #level=3,
                                #csize=5,
                                #threshold=0.6,
                                #wsize=6,
                                #minImageNum=3,
                                #CPU=7,
                                #useVisData=1,
                                #sequence=-1,
                                #timages=None,
                                #oimages=0,
                                #numNeighbors=2),
    'pmvs_medium': PMVS2Options(
        numCameras=12, level=1, csize=4,
        numNeighbors=2)
    #,
    #'pmvs_2_2_1': PMVS2Options(
    #numCameras=12, level=2, csize=2,
    #numNeighbors=1),
    #'pmvs_2_4_1': PMVS2Options(
    #numCameras=12, level=2, csize=4,
    #numNeighbors=1),
    #'pmvs_2_8_1': PMVS2Options(
    #numCameras=12, level=2, csize=8,
    #numNeighbors=1),
    #'pmvs_2_2_2': PMVS2Options(
    #numCameras=12, level=2, csize=2,
    #numNeighbors=2),
    #'pmvs_2_4_2': PMVS2Options(
    #numCameras=12, level=2, csize=4,
    #numNeighbors=2),
    #'pmvs_2_8_2': PMVS2Options(
    #numCameras=12, level=2, csize=8,
    #numNeighbors=2),
    #'pmvs_1_4_2': PMVS2Options(
    #numCameras=12, level=1, csize=4,
    #numNeighbors=2)
    #,
    #'pmvs_0_4_2': PMVS2Options(
    #numCameras=12, level=0, csize=4,
    #numNeighbors=2
    #)  # Used for generating the references (followed by hand cleanup)
}
pmvsOptionNames = pmvsOptionsDict.keys()

if __name__=='__main__':
    print('Attempting to run a reconstruction using pmvs')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_pmvs')
    #options = pmvsOptionsDict['pmvs_2_2_1']
    options = pmvsOptionsDict['pmvs_medium']
    #options = pmvsOptionsDict['pmvs_tuned1']
    run_pmvs(imagesPath, workDirectory=workDirectory, options=options)
    #run_pmvs(imagesPath, options=options) # to test temp directory
