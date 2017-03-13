#!/usr/bin/env python3

# Code for performing reconstruction using openCV 3 through opencv's python bindings.

import numpy
import pathlib
from pathlib import Path
from time import time

import shutil
import glob

from load_ply import save_ply_file

import os
import inspect
pwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

from load_camera_info import load_intrinsics, load_extrinsics

import cv2

# OpenCV has several stereo matchers with incompatible options lists.
# I will define a different options class for each type of options,
# and use the type of the passed options structure to switch between
# stereo matchers.

class StereoBMOptions():
    def __init__(self,
            channels=1,
            alpha = 0.5, # Scaling parameter
            newImageSize = (0,0),
            preset = None,
            numDisparities = 320,
            blockSize=21,
            ):
        assert alpha>=0.0 and alpha <= 1.0, 'Alpha must be in the range [0.0,1.0]'
        self.alpha = alpha
        self.channels = channels
        self.newImageSize = newImageSize
        if preset is None:
            #preset = cv2.STEREO_BM_NARROW_PRESET,
            self.preset = None
        else:
            self.preset = preset
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        assert channels is not None, 'StereoSGBMOptions need to know how many channels there are!'

    def __hash__(self):
        fields = list(self.__dict__.items())
        fields.sort()
        return hash(tuple(fields)) # 

class StereoSGBMOptions():
    def __init__(self,
            channels=1,
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
        self.newImageSize = newImageSize
        if minDisparity is None:
            self.minDisparity = 0*16 # Affects farthest pixel. Set smaller to enable farther.
        else:
            self.minDisparity = minDisparity
        if numDisparities is None:
            maxDisparity = 7*16 # Seems to affect nearest reconstructable pixel. Set larger to enable closer pixels.
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

# Some hard-coded options, roughly slow to fast
opencvOptionsDict = {'sgbm_defaults': StereoSGBMOptions(channels=1), 'bm_defaults':StereoBMOptions(channels=1)}
opencvOptionNames = opencvOptionsDict.keys()

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


class OpenCVStereoMatcher():
    """ Wrapper class that applies OpenCV's stereo matchers pairwise on an array of cameras. 
        If a list of only one instance is passed in matcher_options, it will be used by all camera pairs.
    """
    def __init__(self,
            matcher_options=[opencvOptionsDict['bm_defaults'],],
            num_cameras=12,
            calibrationsPath=None,
            topology='overlapping',
            rectification_interpolation=cv2.INTER_LINEAR,
            visual_debug=False,
            ):
        self.num_cameras = num_cameras
        self.topology = topology
        self.rectification_interpolation=rectification_interpolation
        self.visual_debug = visual_debug

        self.num_pairs = len(topologies)
        if type(matcher_options) is not list:
            self.matcher_options = [matcher_options,]
        if len(self.matcher_options) == 1:
            self.matcher_options *= self.num_cameras

        assert calibrationsPath is not None, 'To initialize an OpenCVStereoMatcher, you must provide a path to the dirctory containing camera intrinsics and extrinsics!'
        self.load_camera_parameters(calibrationsPath)

    def load_images(self,imagesPath):
        # Load a set of images from disk. Doesn't do processing yet.
        imagesPath = imagesPath.resolve()

        # Load the undistorted images off of disk
        print('Loading the images off of disk...')
        num_cameras = len(list(imagesPath.glob('*.png')))
        assert self.num_cameras == num_cameras, 'Mismatch in the number of available images!'
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
            expected_parameters = self.all_camera_parameters[i]
            w,h = expected_parameters[3], expected_parameters[4]
            assert grayImage.shape == (h,w), 'Mismatch in image sizes!'
        self.images = images

    def free_images(self):
        del self.images

    def load_camera_parameters(self, calibrationsPath):
        # Load the camera parameters for every camera in the array
        all_camera_parameters = []
        for i in range(self.num_cameras):
            # Load the intrinsics
            intrinsicsFilePath = calibrationsPath / ('intrinsics_camera%02i.txt' % (i + 1))
            print('Loading intrinsics for camera',i,'from',intrinsicsFilePath,'...')
            assert intrinsicsFilePath.is_file(), "Couldn't find camera intrinsics in "+str(intrinsicsFilePath)
            camera_matrix, distortion_coefficients, image_width, image_height = load_intrinsics(intrinsicsFilePath)
            # The images must already be radially undistorted
            assert(abs(distortion_coefficients[0]) < .000000001)
            assert(abs(distortion_coefficients[1]) < .000000001)
            assert(abs(distortion_coefficients[2]) < .000000001)
            assert(abs(distortion_coefficients[3]) < .000000001)
            assert(abs(distortion_coefficients[4]) < .000000001)

            # Load the extrinsics
            extrinsicsFilePath = calibrationsPath / ('extrinsics_camera%02i.txt' % (i + 1))
            print('Loading extrinsics for camera',i,'from',extrinsicsFilePath,'...')
            R, T = load_extrinsics(extrinsicsFilePath)

            # OpenCV expects the inverse of the transform that HALCON exports!
            R,T = R.T,numpy.dot(-R.T,T)
            all_camera_parameters.append((camera_matrix, R, T, image_width, image_height))
        self.all_camera_parameters = all_camera_parameters

    def run_from_memory(self, images):
        """ Perform stereo reconstruction on a set of images already in memory, and return results in memory. """
        assert self.all_camera_parameters is not None, 'Camera parameters not loaded yet; You should run load_camera_parameters first!'

        t1 = time()
        xyz_global_array = []
        for left_index,right_index in topologies[self.topology]:
            print('Performing Stereo matching between cameras', left_index,'and',right_index,'...')
            left_image, right_image = self.images[left_index], self.images[right_index]

            left_camera_matrix, left_R, left_T, left_width, left_height = self.all_camera_parameters[left_index]
            right_camera_matrix, right_R, right_T, right_width, right_height = self.all_camera_parameters[right_index]
            assert left_width == right_width, "Images of mismatched resolution is untested!"
            assert left_height == right_height, "Images of mismatched resolution is untested!"
            h,w = left_height, left_width

            # TODO: use pyrDown to support downsampling the images by factors of two?

            # Perform rectification; this is shared by OpenCV's algorithms
            flags=0
            #flags=cv2.CALIB_ZERO_DISPARITY

            dist_coefs = (0.0,0.0,0.0,0.0,0.0)
            imageSize = (w, h)
            if options.newImageSize == (0,0):
                options.newImageSize = imageSize

            # Form the transformation between the two camera frames, needed for stereoRectify.
            R_intercamera = numpy.dot(right_R, left_R.T)
            T_intercamera = right_T - numpy.dot(R_intercamera, left_T)

            left_R_rectified, right_R_rectified, P1_rect, P2_rect, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                cameraMatrix1 = left_camera_matrix,
                distCoeffs1 = dist_coefs,
                cameraMatrix2 = right_camera_matrix,
                distCoeffs2 = dist_coefs,
                imageSize=imageSize,
                newImageSize=options.newImageSize,
                R=R_intercamera,
                T=T_intercamera,
                flags=flags,
                alpha=options.alpha)
            # Geometry note: left_R_rectified above is apparently the rotation that does rectification, i.e
            #   something close to an identity matrix, NOT the new transformation back to global coordinates.

            # Create rectification maps
            rectification_map_type = cv2.CV_16SC2
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

            # Instantiate the matchers; they may do something slow internally...
            matching_options = options.__dict__.copy()
            del matching_options['channels']
            del matching_options['newImageSize']
            del matching_options['alpha']
            if type(options)==StereoSGBMOptions:
                # Perform stereo matching using SGBM
                create_matcher = cv2.StereoSGBM_create
                matcher = create_matcher(**matching_options)
            elif type(options) == StereoBMOptions:
                # Perform stereo matching using normal block matching
                create_matcher = cv2.StereoBM_create
                del matching_options['preset']
                matcher = create_matcher(**matching_options)

            # Apply the rectification maps
            left_image_rectified = cv2.remap(left_image, left_maps[0],
                                             left_maps[1], self.rectification_interpolation)
            right_image_rectified = cv2.remap(right_image, right_maps[0],
                                              right_maps[1], self.rectification_interpolation)

            #if self.visual_debug:
            #left = numpy.array(left_image_rectified)
            #right = numpy.array(right_image_rectified)
            #leftright = numpy.hstack((left,right))
            #pylab.imshow(leftright)
            #pylab.show()
            #return
            #continue

            disparity_image = matcher.compute(left_image_rectified, right_image_rectified)
            # WARNING! OpenCV 3 Apparently doesn't support floating point disparity anymore,
            # and 16 bit disparity needs to be divided by 16
            if disparity_image.dtype == numpy.int16:
                disparity_image = disparity_image.astype(numpy.float32)
                disparity_image /= 16

            if self.visual_debug:
                im = numpy.array(disparity_image)
                #pylab.imshow(numpy.vstack((left_image,im,right_image)))
                pylab.imshow(numpy.vstack((im,)))
                pylab.show()
                #continue

                # Convert the depth map to a point cloud
            threedeeimage = cv2.reprojectImageTo3D(disparity_image, Q, handleMissingValues=True,ddepth=cv2.CV_32F)
            threedeeimage = numpy.array(threedeeimage)
            #if self.visual_debug:
            #depth_image = threedeeimage[:,:,2]
            #pylab.imshow(depth_image)
            #pylab.show()
            #continue

            # Put the 3D images in a unified coordinate system...
            xyz = threedeeimage.reshape((h*w,3)) # x,y,z now in three columns, in left rectified camera coordinates

            z = xyz[:,2]
            goodz = z < 1e3
            xyz_filtered = xyz[goodz,:]
            #print('pixels before filtering: ',h*w, "after filtering:" ,xyz_filtered.shape[0] )

            R2,T2 = left_R, left_T # perspective is from left image.
            R3,T3 = R2.T,numpy.dot(-R2.T,T2) # Invert direction of transformation to map camera to world. correct
            R_left_rectified_to_global = numpy.dot(R3,left_R_rectified.T)
            xyz_global = numpy.dot(xyz_filtered, R_left_rectified_to_global.T) + T3.T  # TODO: combine this with the the multipilication by Q inside of reprojectImageTo3D above. Note that different filtering may be required.

            #save_ply_file(xyz_global, 'pair_'+str(left_index)+'_'+str(right_index)+'.ply')
            xyz_global_array.append(xyz_global)

        xyz = numpy.vstack(xyz_global_array)
        t2 = time()
        dt = t2-t1 # seconds. 
        
        return xyz, dt

    def run_from_disk(self,
                      imagesPath,
                      calibrationsPath=None,
                      destDir=None,
                      destFile=None,
                      options = opencvOptionsDict['bm_defaults'],
                      workDirectory=None,
                      runtimeFile=None):
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
        if self.visual_debug:
            import pylab

        self.load_images(imagesPath)

        if calibrationsPath is not None:
            self.load_camera_parameters(calibrationsPath)

        # By default, work in a temporary directory.
        # "with...as" ensures the temp directory is cleared even if there is an error below.
        if workDirectory is None:
            from tempfile import TemporaryDirectory
            workDirectory = TemporaryDirectory(dir=str(Path(pwd)/'tmp'))
        if not workDirectory.is_dir():
            workDirectory.mkdir()

        xyz, reconstruction_time = self.run_from_memory(self.images)
        del self.images

        ## Copy the file to the appropriate destination
        if destDir is None:
            destDir = Path.cwd()
        if destFile is None:
            destFile = 'reconstruction.ply'
            destPath = destDir / destFile

        print('Saving reconstruction to',destPath,'...')
        save_ply_file(xyz, destPath)

        if runtimeFile is None:
            runtimeFile = destPath.parent / (destPath.stem +'_runtime.txt')
        with open(str(runtimeFile), 'w') as fd:
            fd.write(str(reconstruction_time)) # seconds


def run_opencv(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None, visual_debug=False):
    """ An inefficent, but simple interface that doesn't precompute anything, and loads from disk. """
    matcher = OpenCVStereoMatcher(matcher_options=options, calibrationsPath=imagesPath, visual_debug=visual_debug)
    matcher.run_from_disk(imagesPath=imagesPath,
                          destDir=destDir,
                          destFile=destFile,
                          workDirectory=workDirectory,
                          runtimeFile=runtimeFile)


if __name__=='__main__':
    print('Attempting to run a reconstruction using opencv')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_opencv')
    #options = opencvOptionsDict['sgbm_defaults'] # Still generating ludicrously large cloud. filtering broken.
    options = opencvOptionsDict['bm_defaults']
    run_opencv(imagesPath=imagesPath, workDirectory=workDirectory, options=options, visual_debug=False)
    #run_opencv(imagesPath, options=options) # to test temp directory
