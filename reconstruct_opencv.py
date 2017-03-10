#!/usr/bin/env python3

# Code for performing reconstruction using openCV

import numpy
import pathlib
from pathlib import Path
from time import time

from load_ply import save_ply_file

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
            channels = None,
            topology='overlapping',
            rectification_interpolation=cv2.INTER_LINEAR,
            alpha = 0.5, # Scaling parameter
            newImageSize = (0,0),
            preset = None,
            numDisparities = 320,
            blockSize=21,
            ):
        self.num_cameras = num_cameras
        self.topology = topology
        self.rectification_interpolation=rectification_interpolation
        assert alpha>=0.0 and alpha <= 1.0, 'Alpha must be in the range [0.0,1.0]'
        self.alpha = alpha
        self.newImageSize = newImageSize
        if preset is None:
            #preset = cv2.STEREO_BM_NARROW_PRESET,
            self.preset = None
        else:
            self.preset = preset
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        assert channels is not None, 'StereoSGBMOptions need to know how many channels there are!'
        self.channels = channels

    def __hash__(self):
        fields = list(self.__dict__.items())
        fields.sort()
        return hash(tuple(fields)) # 

class StereoSGBMOptions():
    def __init__(self,
            num_cameras=12,
            channels=None,
            topology='overlapping',
            rectification_interpolation=cv2.INTER_LINEAR,
            alpha = 0.5, # Scaling parameter
            newImageSize = (0,0),
            minDisparity = None,
            numDisparities = None,
            blockSize = 11, # Must be odd! normally 3..11
            P1 = None,
            P2 = None,
            disp12MaxDiff=-1, # <0 to disable left-right check. Bigger to allow more left/right inconsistency
            preFilterCap=64, # Smaller is stricter?
            uniquenessRatio=10, # Percent. Normally between 5-15. Seems to work [0,30] Set Bigger to throw out more pixels
            speckleWindowSize=000, # 0 to disable speckle filtering
            speckleRange=32, # disparity within component / 16. Normally 1 or 2
            mode = (cv2.StereoSGBM_MODE_SGBM, cv2.StereoSGBM_MODE_HH, cv2.StereoSGBM_MODE_SGBM_3WAY)[0]
            ):
        self.num_cameras = num_cameras
        self.topology = topology
        self.rectification_interpolation=rectification_interpolation
        assert alpha>=0.0 and alpha <= 1.0, 'Alpha must be in the range [0.0,1.0]'
        self.alpha = alpha
        self.newImageSize = newImageSize
        if minDisparity is None:
            self.minDisparity = 0*16 # Affects farthest pixel. Set smaller to enable farther.
        else:
            self.minDisparity = minDisparity
        if numDisparities is None:
            maxDisparity = 7*16 # Seems to affect nearest reconstructable pixel. Set larger to enable closer pixels.
            #self.numDisparities = 16*60 # 960
            #self.numDisparities = 16*11 # 176
            #self.numDisparities = 16*5 # 176
            self.numDisparities = maxDisparity - self.minDisparity
            self.numDisparities -= self.numDisparities%16
        else:
            self.numDisparities = numDisparities
        assert self.numDisparities%16==0, 'numDisparities must be a multiple of 16!'
        self.blockSize = blockSize
        assert self.blockSize % 2 == 1, 'blockSize must be odd!'
        assert channels is not None, 'StereoSGBMOptions need to know how many channels there are!'
        self.channels = channels
        if P1 is None:
            self.P1 = 4*channels*blockSize**2
        else:
            self.P1 = P1
        if P2 is None:
            self.P2 = 16*channels*blockSize**2
        else:
            self.P2 = P2
        assert self.P2>self.P1, 'Smoothness Parameters must satisfy P2>P1!'
        self.disp12MaxDiff=disp12MaxDiff
        self.preFilterCap=preFilterCap
        self.uniquenessRatio=uniquenessRatio
        self.speckleWindowSize=speckleWindowSize
        self.speckleRange=speckleRange
        self.mode = mode

    # Reasonable stereo matching topologies for a 12 camera ring. Note, there is an additional choice of whether to match in both directions. 
    # Note that whether this is (left,right) or (right,left) depends on whether the cameras are right side up or upside down!
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


def run_opencv(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None, VISUAL_DEBUG=False):
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
    if VISUAL_DEBUG:
        import pylab

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
                     workDirectory=Path(workDirectory),
                     VISUAL_DEBUG=VISUAL_DEBUG)
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

        # OpenCV expects the inverse of the transform that HALCON exports!
        R,T = R.T,numpy.dot(-R.T,T)
        all_camera_parameters.append((cameraMatrix, R, T))

    threedeeimages = []
    # Run OpenCV on the pairs of images
    t1 = time()
    #for right_index,left_index in topologies[options.topology]: # doesn't work
    for left_index,right_index in topologies[options.topology]:
        left_image, right_image = images[left_index], images[right_index]
        left_camera_matrix, left_R, left_T = all_camera_parameters[left_index]
        right_camera_matrix, right_R, right_T = all_camera_parameters[right_index]

        # Perform rectification; this is shared by OpenCV's algorithms
        flags=0
        #flags=cv2.CALIB_ZERO_DISPARITY
        h, w = left_image.shape
        #print('h =',h,'w =',w)
        dist_coefs = (0.0,0.0,0.0,0.0,0.0)
        imageSize = (w, h)
        if options.newImageSize == (0,0):
            options.newImageSize = imageSize

        # For x1 = left_R*x0+left_T interpretation
        R = numpy.dot(right_R, left_R.T)
        T = right_T - numpy.dot(R, left_T)

        # For x0 = left_R*x1+left_T interpretation
        #R = numpy.dot(left_R.T,right_R)
        #T = numpy.dot(left_R.T,right_T-left_T)

        left_R_rectified, right_R_rectified, P1_rect, P2_rect, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            cameraMatrix1 = left_camera_matrix,
            distCoeffs1 = dist_coefs,
            cameraMatrix2 = right_camera_matrix,
            distCoeffs2 = dist_coefs,
            imageSize=imageSize,
            newImageSize=options.newImageSize,
            R=R,
            T=T,
            flags=flags,
            alpha=options.alpha)

        # Create rectification maps
        rectification_map_type = cv2.CV_16SC2 
        #rectification_map_type = cv2.CV_32F 
        left_maps = cv2.initUndistortRectifyMap(left_camera_matrix,
                                                dist_coefs,
                                                left_R_rectified,
                                                P1_rect,
                                                size=options.newImageSize,
                                                m1type=rectification_map_type)
        right_maps = cv2.initUndistortRectifyMap(right_camera_matrix,
                                                 dist_coefs,
                                                 right_R_rectified,
                                                 P2_rect,
                                                 size=options.newImageSize,
                                                 m1type=rectification_map_type)

        # Apply the rectification maps
        left_image_rectified = cv2.remap(left_image, left_maps[0],
                                         left_maps[1], options.rectification_interpolation)
        right_image_rectified = cv2.remap(right_image, right_maps[0],
                                          right_maps[1], options.rectification_interpolation)

        #if VISUAL_DEBUG:
        #left = numpy.array(left_image_rectified)
        #right = numpy.array(right_image_rectified)
        #leftright = numpy.hstack((left,right))
        #pylab.imshow(leftright)
        #pylab.show()
        #return
        #continue
        matching_options = options.__dict__.copy()
        del matching_options['topology']
        del matching_options['num_cameras']
        del matching_options['channels']
        del matching_options['newImageSize']
        del matching_options['alpha']
        del matching_options['rectification_interpolation']

        if type(options)==StereoSGBMOptions:
            # Perform stereo matching using SGBM
            create_matcher = cv2.StereoSGBM_create
            matcher = create_matcher(**matching_options)
        elif type(options) == StereoBMOptions:
            # Perform stereo matching using normal block matching
            create_matcher = cv2.StereoBM_create
            del matching_options['preset']
            print(matching_options)
            matcher = create_matcher(**matching_options)
        print('computing disparity...')

        disparity_image = matcher.compute(left_image_rectified, right_image_rectified)
        # WARNING! OpenCV 3 Apparently doesn't support floating point disparity anymore,
        # and 16 bit disparity needs to be divided by 16
        if disparity_image.dtype == numpy.int16:
            disparity_image = disparity_image.astype(numpy.float32)
            disparity_image /= 16

        if VISUAL_DEBUG:
            im = numpy.array(disparity_image)
            #pylab.imshow(numpy.vstack((left_image,im,right_image)))
            pylab.imshow(numpy.vstack((im,)))
            pylab.show()
            #continue

        # Convert the depth map to a point cloud
        print('Q.dtype=',Q.dtype)
        print(Q)
        
        threedeeimage = cv2.reprojectImageTo3D(disparity_image, Q, handleMissingValues=True,ddepth=cv2.CV_32F)
        #threedeeimage = cv2.reprojectImageTo3D(disparity_image, Q, handleMissingValues=True)
        threedeeimage = numpy.array(threedeeimage)
        #if VISUAL_DEBUG:
            #depth_image = threedeeimage[:,:,2]
            #pylab.imshow(depth_image)
            #pylab.show()
            #continue

        # Put the 3D images in a unified coordinate system...
        xyz = threedeeimage.reshape((h*w,3)) # x,y,z now in three columns

        z = xyz[:,2]
        goodz = z < 1e3
        xyz_filtered = xyz[goodz,:]
        print('pixels before filtering: ',h*w, "after filtering:" ,xyz_filtered.shape[0] )

        # If perspective is from left image:
        R,T = left_R, left_T
        #R,T = right_R, right_T
        #R,T = left_R_rectified, left_T  # nope
        #R,T = left_R_rectified*left_R, left_T  # nope
        #R,T = right_R_rectified, right_T 

        xyz_filtered = numpy.dot(xyz_filtered, left_R_rectified) # NO IDEA WHY THIS IS NECESSARY! WTF?

        R,T = R.T,numpy.dot(-R.T,T) # Invert direction of transformation to map camera to world. correct
        
        #xyz_global = xyz_filtered

        xyz_filtered[0,:] = 0.0 # Debug: should make the camera centers visible.
        xyz_filtered[1,:] = [0,0,0.005] # Debug: should make the camera direction visible
        xyz_filtered[2,:] = [0,0,0.03] # Debug: should make the camera direction visible.

        xyz_global = numpy.dot(xyz_filtered, R.T) + T.T # Transposing because of right multiply
        save_ply_file(xyz_global, 'pair_'+str(left_index)+'_'+str(right_index)+'.ply')

    t2 = time()
    dt = t2-t1 # seconds. 
    return

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
        assert False, 'a .ply file was not generated; reconstruction must have failed!'


# Some hard-coded options, roughly slow to fast
opencvOptionsDict = {'sgbm_defaults': StereoSGBMOptions(channels=1), 'bm_defaults':StereoBMOptions(channels=1)}
opencvOptionNames = opencvOptionsDict.keys()

if __name__=='__main__':
    print('Attempting to run a reconstruction using opencv')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_opencv')
    #options = opencvOptionsDict['sgbm_defaults']
    options = opencvOptionsDict['bm_defaults']
    run_opencv(imagesPath, workDirectory=workDirectory, options=options, VISUAL_DEBUG=False)
    #run_opencv(imagesPath, options=options) # to test temp directory
