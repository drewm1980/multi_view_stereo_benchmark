#!/usr/bin/env python3

# Code for performing reconstruction using openCV

import numpy
import pathlib
from pathlib import Path

import os
import inspect
pwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
#openCVPath = Path(pwd) / 'extern/CMVS-PMVS/program/main/openCV'
#assert openCVPath.is_file(), "openCV binary not found. Try running bootstrap.sh?"

from load_camera_info import load_intrinsics, load_extrinsics

import cv2
#cv2.startWindowThread()
#cv2.destroyAllWindows()

# OpenCV has several stereo matchers with incompatible options lists.
# I will define a different options class for each type of options,
# and use the type of the passed options structure to switch between
# stereo matchers.

class StereoBMOptions():
    def __init__(self, 
            num_cameras=12,
            topology='overlapping',
            rectification_interpolation=cv2.INTER_LINEAR,
            preset = None,
            ndisparities = 320,
            SADWindowSize=21,
            ):
        self.num_cameras = num_cameras
        self.topology = topology
        self.rectification_interpolation=rectification_interpolation
        if preset is None:
            #preset = cv2.STEREO_BM_NARROW_PRESET,
            self.preset = None
        else:
            self.preset = preset
        self.ndisparities = ndisparities
        self.SADWindowSize = SADWindowSize

    def __hash__(self):
        fields = list(self.__dict__.items())
        fields.sort()
        return hash(tuple(fields)) # 

class StereoSGBMOptions():
    def __init__(self,
            num_cameras=12,
            topology='overlapping',
            minDisparity = -1000,
            numDisparities = None,
            SADWindowSize = 16,
            P1 = None,
            P2 = None,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
            ):
        self.num_cameras = num_cameras
        self.topology = topology
        self.minDisparity = minDisparity
        if numDisparities is None:
            self.numDisparities = 1000 - self.minDisparity
        self.SADWindowSize = SADWindowSize
        if P1 is None:
            P1 = 8*3*SADWindowSize**2
        else:
            self.P1 = P1
        if P2 is None:
            P2 = 32*3*SADWindowSize**2
        else:
            self.P2 = P2
        self.P2 = P2
        self.disp12MaxDiff=disp12MaxDiff,
        self.uniquenessRatio=uniquenessRatio
        self.speckleWindowSize=speckleWindowSize
        self.speckleRange=speckleRange

    # Reasonable stereo matching topologies for a 12 camera ring. Note, there is an additional choice of whether to match in both directions.
import collections
topologies = collections.OrderedDict()
topologies['overlapping'] = tuple(zip((0,1,2,3,4,5,6,7,8,9,10,11),
                          (1,2,3,4,5,6,7,8,9,10,11,0)))
topologies['adjacent'] = tuple(zip((0,2,4,6,8,10),
                     (1,3,5,7,9,11)))
topologies['skipping_1'] = tuple(zip((0,3,6,9),
                 (1,4,7,10)))
topologies['skipping_2'] = tuple(zip((0,4,8),
                 (1,5,9)))


def run_opencv(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None):
    """ Run OpenCV's stereo matcher on a directory full of images.

        The images must ALREADY be radially undistorted!

    Arguments:
    imagesPath -- A directory full of source images
    destDir -- The destination directory of the ply file. (default current directory)
    destFile -- The destination name of the ply file. (default <name of the directory>.ply)
    options -- An instance of OpenCVOptions
    workDirectory -- Existing directory where intermediate results may be written for debugging. (default generates a temp directory)
    runtimeFile -- The name of a file where info regarding the runtime will be stored.
    """
    import shutil
    import glob

    # By default, work in a temporary directory.
    # "with...as" ensures the temp directory is cleared even if there is an error below.
    if workDirectory is None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(dir=str(Path(pwd)/'tmp')) as workDirectory:
            run_opencv(imagesPath=imagesPath,
                     destDir=destDir,
                     destFile=destFile,
                     options=options,
                     runtimeFile=runtimeFile,
                     workDirectory=Path(workDirectory))
        return
    if not workDirectory.is_dir():
        workDirectory.mkdir()

    imagesPath = imagesPath.resolve()

    # Load the undistorted images off of disk
    print('Loading the images off of disk...')
    num_cameras = len(list(imagesPath.glob('*.png')))
    assert num_cameras == options.num_cameras, 'Mismatch in the number of available images!'
    images = []
    for i in range(num_cameras):
        fileName = 'image_camera%02i.png' % (i + 1)
        filePath = imagesPath / fileName
        print('Loading image',filePath)
        colorImage = cv2.imread(str(filePath))
        grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
        images.append(grayImage)
        #cv2.namedWindow("grayImage",cv2.WINDOW_NORMAL)
        #cv2.imshow("grayImage", grayImage)
        #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Load the camera parameters
    all_camera_parameters = []
    for i in range(num_cameras):
        # Load the intrinsics
        intrinsicsFilePath = imagesPath / ('intrinsics_camera%02i.txt' % (i + 1))
        print('Loading intrinsics for camera',i,'from',intrinsicsFilePath,'...')
        assert intrinsicsFilePath.is_file(), "Couldn't find camera intrinsics in "+str(intrinsicsFilePath)
        cameraMatrix, distCoffs = load_intrinsics(intrinsicsFilePath)
        # The images must already be radially undistorted
        assert(abs(distCoffs[0]) < .000000001)
        assert(abs(distCoffs[1]) < .000000001)
        assert(abs(distCoffs[2]) < .000000001)
        assert(abs(distCoffs[3]) < .000000001)
        assert(abs(distCoffs[4]) < .000000001)

        # Load the extrinsics
        extrinsicsFilePath = imagesPath / ('extrinsics_camera%02i.txt' % (i + 1))
        print('Loading extrinsics for camera',i,'from',extrinsicsFilePath,'...')
        R, T = load_extrinsics(extrinsicsFilePath)

        # OpenCV probably expects the inverse of the transform that HALCON exports!
        R,T = R.T,numpy.dot(-R.T,T)
        all_camera_parameters.append((cameraMatrix, R, T))

    # Run OpenCV on the pairs of images
    t1 = time()
    for left_index,right_index in topologies[options.topology]:
        left_image, right_image = images[left_index], images[right_index]
        left_camera_matrix, left_R, left_T = all_camera_parameters[left_index]
        right_camera_matrix, right_R, right_T = all_camera_parameters[right_index]

        # Perform rectification; this is shared by OpenCV's algorithms

        if type(options)==StereoSGBMOptions:
            # Perform stereo matching using SGBM
            pass
        elif type(options) == StereoBMOptions:
            # Perform stereo matching using normal block matching
            pass

        # Convert the depth map to a point cloud

    
    t2 = time()
    dt = t2-t1 # seconds. 

    ## Copy the file to the appropriate destination
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
opencvOptionsDict = {'sgbm_defaults': StereoSGBMOptions(), 'bm_defaults':StereoBMOptions()}
opencvOptionNames = opencvOptionsDict.keys()

if __name__=='__main__':
    print('Attempting to run a reconstruction using opencv')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_opencv')
    options = opencvOptionsDict['sgbm_defaults']
    run_opencv(imagesPath, workDirectory=workDirectory, options=options)
    #run_opencv(imagesPath, options=options) # to test temp directory
