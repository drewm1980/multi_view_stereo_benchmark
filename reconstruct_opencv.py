#!/usr/bin/env python3

# Code for performing reconstruction using openCV 3 through opencv's python bindings.

import numpy
import pathlib
from pathlib import Path
from time import time

import shutil
import glob

from load_ply import save_ply

import os
import inspect
pwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

from load_camera_info import load_intrinsics, load_extrinsics

import cv2

# OpenCV has several stereo matchers with incompatible options lists.
# I will define a different options class for each type of options,
# and use the type of the passed options structure to switch between
# stereo matchers.

# Taken from the canonical exported API definition in calib3d.hpp from Opencv 3.2.0 in 
# nix: /nix/store/6x2783a5k8p75d1klspqqn6dwlg1j8mb-opencv-3.2.0-src/modules/calib3d/include/opencv2
StereoRectifyOptions = {'imageSize':(1280,1024),
                        'flags':(0,cv2.CALIB_ZERO_DISPARITY)[0], # TODO explore other flags
                        'newImageSize':(1280,1024),
                        'alpha':0.5}
StereoMatcherOptions = {'MinDisparity': 10,
                        'NumDisparities': 320,
                        'BlockSize': 21,
                        'SpeckleWindowSize': 0,
                        'SpeckleRange': 0,
                        'Disp12MaxDiff': 0}
StereoBMOptions = {
        'PreFilterType': (cv2.StereoBM_PREFILTER_NORMALIZED_RESPONSE, cv2.StereoBM_PREFILTER_XSOBEL)[0],
                   'PreFilterSize': 5, # preFilterSize must be odd and be within 5..255
                   'PreFilterCap': 63, # preFilterCap must be within 1..63. Used to truncate pixel values
                   'TextureThreshold': 10,
                   'UniquenessRatio': 10,
                   'SmallerBlockSize': 16 * 5,
                   #'ROI1', # I don't really want to set these
                   #'ROI2'
                   }
StereoSGBMOptions = {'PreFilterCap': 0,
                     'UniquenessRatio': 0,
                     'P1': 10,
                     'P2': 10,
                     'Mode': (cv2.StereoSGBM_MODE_SGBM, cv2.StereoSGBM_MODE_HH,
                              cv2.StereoSGBM_MODE_SGBM_3WAY)[0]}
RemapOptions = {'interpolation':cv2.INTER_LINEAR}
CameraArrayOptions = {
                        'channels':1,
                        'num_cameras':12,
                        'topology':'adjacent'}

DefaultOptionsBM = {'StereoRectify':StereoRectifyOptions,
        'StereoMatcher':StereoMatcherOptions,
        'StereoBM':StereoBMOptions,
        'CameraArray':CameraArrayOptions,
        'Remap':RemapOptions}


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
_search_space = nested_dict_to_list_of_tuples(DefaultOptionsBM, extra_nesting=True)


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


# Some hard-coded options, roughly slow to fast
opencvOptionsDict = {'opencv_block_matcher_defaults': DefaultOptionsBM}
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
        Usage: Re-instantiate each time the camera geometry changes with a new calibrationsPath.
            For each reconstruction, call either run_from_memory or run_from_disk depending on your use case.
    """
    def __init__(self,
            options=DefaultOptionsBM,
            calibrationsPath=None,
            visual_debug=False,
            ):
        self.options = options
        self.num_cameras = options['CameraArray']['num_cameras']
        self.topology = options['CameraArray']['topology']
        self.visual_debug = visual_debug

        assert calibrationsPath is not None, 'To initialize an OpenCVStereoMatcher, you must provide a path to the dirctory containing camera intrinsics and extrinsics!'
        self.load_camera_parameters(calibrationsPath)

        self.left_maps_array = []
        self.right_maps_array = []
        self.matchers = []
        self.Q_array = []
        self.extrinsics_left_rectified_to_global_array = []

        for pair_index, (left_index,right_index) in enumerate(topologies[self.topology]):
            left_camera_matrix, left_R, left_T, left_width, left_height = self.all_camera_parameters[left_index]
            right_camera_matrix, right_R, right_T, right_width, right_height = self.all_camera_parameters[right_index]
            assert left_width == right_width, "Images of mismatched resolution is unsupported by opencv!"
            assert left_height == right_height, "Images of mismatched resolution is unsupported by opencv!"
            h,w = left_height, left_width

            # TODO: use pyrDown to support downsampling the images by factors of two?

            # Perform rectification; this is shared by OpenCV's algorithms
            flags=options['StereoRectify']['flags']

            distortion_coefficients = (0.0,0.0,0.0,0.0,0.0) # TODO extend to unrectified images
            left_distortion_coefficients = distortion_coefficients
            right_distortion_coefficients = distortion_coefficients
            imageSize = options['StereoRectify']['imageSize'] # w,h
            newImageSize = options['StereoRectify']['newImageSize']
            alpha = options['StereoRectify']['alpha']

            # Form the transformation between the two camera frames, needed for stereoRectify.
            R_intercamera = numpy.dot(right_R, left_R.T)
            T_intercamera = right_T - numpy.dot(R_intercamera, left_T)

            left_R_rectified, right_R_rectified, P1_rect, P2_rect, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                cameraMatrix1 = left_camera_matrix,
                distCoeffs1 = left_distortion_coefficients,
                cameraMatrix2 = right_camera_matrix,
                distCoeffs2 = right_distortion_coefficients,
                imageSize=imageSize,
                newImageSize=newImageSize,
                R=R_intercamera,
                T=T_intercamera,
                flags=flags,
                alpha=alpha)

            self.Q_array.append(Q)

            # Geometry note: left_R_rectified above is apparently the rotation that does rectification, i.e
            #   something close to an identity matrix, NOT the new transformation back to global coordinates.

            # Some geometry needed for converting disparity back to global coordinates
            R2,T2 = left_R, left_T # perspective is from left image.
            R3,T3 = R2.T,numpy.dot(-R2.T,T2) # Invert direction of transformation to map camera to world. 
            R_left_rectified_to_global = numpy.dot(R3,left_R_rectified.T)
            T_left_rectified_to_global = T3
            extrinsics_left_rectified_to_global = R_left_rectified_to_global.astype(numpy.float32), T_left_rectified_to_global.astype(numpy.float32)
            self.extrinsics_left_rectified_to_global_array.append(extrinsics_left_rectified_to_global)

            # Create rectification maps
            rectification_map_type = cv2.CV_16SC2
            left_maps = cv2.initUndistortRectifyMap(left_camera_matrix,
                                                    left_distortion_coefficients,
                                                    left_R_rectified,
                                                    P1_rect,
                                                    size=newImageSize,
                                                    m1type=rectification_map_type)
            right_maps = cv2.initUndistortRectifyMap(right_camera_matrix,
                                                     right_distortion_coefficients,
                                                     right_R_rectified,
                                                     P2_rect,
                                                     size=newImageSize,
                                                     m1type=rectification_map_type)
            self.left_maps_array.append(left_maps)
            self.right_maps_array.append(right_maps)

            # Instantiate the matchers; they may do something slow internally...
            if 'StereoBM' in options:
                # Perform stereo matching using normal block matching
                numDisparities = options['StereoMatcher']['NumDisparities']
                blockSize = options['StereoMatcher']['BlockSize']
                matcher = cv2.StereoBM_create(numDisparities=numDisparities,blockSize=blockSize)
                setterOptions = {}
                setterOptions.update(options['StereoMatcher'])
                setterOptions.update(options['StereoBM'])
                for key,value in setterOptions.items():
                    setter = eval('matcher.set'+key) # Returns the setter function
                    setter(value) # Calls the setter function.
            elif 'StereoSGBM' in options:
                # Perform stereo matching using SGBM
                minDisparity = options['StereoMatcher']['MinDisparity']
                numDisparities = options['StereoMatcher']['NumDisparities']
                blockSize = options['StereoMatcher']['BlockSize']
                matcher = cv2.StereoSGBM_create(minDisparity=minDisparity,
                                            numDisparities=numDisparities,
                                            blockSize=blockSize)
                setterOptions = {}
                setterOptions.update(options['StereoMatcher'])
                setterOptions.update(options['StereoSGBM'])
                for key,value in setterOptions.items():
                    setter = eval('matcher.set'+key) # Returns the setter function
                    setter(value) # Calls the setter function.
            else:
                assert False, "Couldn't determine the matcher type from passed options!"
            self.matchers.append(matcher)


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
        for pair_index, (left_index,right_index) in enumerate(topologies[self.topology]):
            print('Performing Stereo matching between cameras', left_index,'and',right_index,'...')
            left_image, right_image = images[left_index], images[right_index]


            left_maps = self.left_maps_array[pair_index]
            right_maps = self.right_maps_array[pair_index]

            # Apply the rectification maps
            remap_interpolation = self.options['Remap']['interpolation']
            left_image_rectified = cv2.remap(left_image, left_maps[0],
                                             left_maps[1], remap_interpolation)
            right_image_rectified = cv2.remap(right_image, right_maps[0],
                                              right_maps[1], remap_interpolation)

            #if self.visual_debug:
            #left = numpy.array(left_image_rectified)
            #right = numpy.array(right_image_rectified)
            #leftright = numpy.hstack((left,right))
            #pylab.imshow(leftright)
            #pylab.show()
            #return
            #continue
            matcher = self.matchers[pair_index]
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
            Q = self.Q_array[pair_index]
            threedeeimage = cv2.reprojectImageTo3D(disparity_image, Q, handleMissingValues=True,ddepth=cv2.CV_32F)
            threedeeimage = numpy.array(threedeeimage)
            #if self.visual_debug:
            #depth_image = threedeeimage[:,:,2]
            #pylab.imshow(depth_image)
            #pylab.show()
            #continue

            # Put the 3D images in a unified coordinate system...
            xyz = threedeeimage.reshape((-1,3)) # x,y,z now in three columns, in left rectified camera coordinates

            z = xyz[:,2]
            goodz = z < 1e3
            xyz_filtered = xyz[goodz,:]
            #print('pixels before filtering: ',h*w, "after filtering:" ,xyz_filtered.shape[0] )

            R_left_rectified_to_global, T_left_rectified_to_global = self.extrinsics_left_rectified_to_global_array[pair_index]
            xyz_global = numpy.dot(xyz_filtered, R_left_rectified_to_global.T) + T_left_rectified_to_global.T  # TODO: combine this with the the multipilication by Q inside of reprojectImageTo3D above. Note that different filtering may be required.

            #save_ply(xyz_global, 'pair_'+str(left_index)+'_'+str(right_index)+'.ply')
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
                      options = opencvOptionsDict['opencv_block_matcher_defaults'],
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
        save_ply(xyz, destPath)

        if runtimeFile is None:
            runtimeFile = destPath.parent / (destPath.stem +'_runtime.txt')
        with open(str(runtimeFile), 'w') as fd:
            fd.write(str(reconstruction_time)) # seconds


def run_opencv(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None, visual_debug=False):
    """ An inefficent, but simple interface that doesn't precompute anything, and loads from disk. """
    matcher = OpenCVStereoMatcher(options=options, calibrationsPath=imagesPath, visual_debug=visual_debug)
    matcher.run_from_disk(imagesPath=imagesPath,
                          destDir=destDir,
                          destFile=destFile,
                          workDirectory=workDirectory,
                          runtimeFile=runtimeFile)


if __name__=='__main__':
    print('Attempting to run a reconstruction using opencv')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_opencv')
    options = DefaultOptionsBM
    run_opencv(imagesPath=imagesPath, workDirectory=workDirectory, options=options, visual_debug=False)
    #run_opencv(imagesPath, options=options) # to test temp directory
