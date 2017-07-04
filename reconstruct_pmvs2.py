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
from .load_ply import load_ply

def set_up_visualize_subdirectory(images=None, inputPath=None, destPath=None):
    """
    Create the "visualize" subdirectory required by PMVS.
        This directory contains the actual source images.
    Inputs:
    inputPath -- full path to a directory containing undistorted images
    destPath -- full path to a directory where the "visualize" subdir will be 
                created
    """
    assert destPath is not None, "destPath is a required argument!"
    assert (images is None)!=(inputPath is None), 'Please pass either "images" or "inputPath" but not both!'
    visualizePath = destPath / 'visualize'
    if not visualizePath.is_dir():
        visualizePath.mkdir()
    print('Setting up visualize subdirectory in ' + str(visualizePath)+'...')
    if images is not None:
        from PIL import Image
        for i,image in enumerate(images):
            assert len(image.shape) in (2,3), 'image shape does not make sense for an image!'
            # Single channel image, needs to be converted to color image or PMVS2 will complain!
            if len(image.shape)==2 or image.shape[2]==1:
                import cv2
                color_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            else:
                assert image.shape[2]==3, 'image has more than one but not 3 channels!'
                color_image = image
            # PMVS takes .ppm files... maybe png too, but we don't want compression overhead
            destFilename = "%08i.ppm"%(i)
            destFilePath = visualizePath / destFilename
            Image.fromarray(color_image).save(destFilePath)
    else:
        # use imageMagick to convert the format of the files to .ppm
        import glob
        numCameras = len(list((inputPath.glob("*.png"))))
        for i in range(numCameras):
            sourceFilename = "image_camera%02i.png"%(i+1)
            destFilename = "%08i.ppm"%(i)
            sourcePath = inputPath / sourceFilename
            destFilePath = visualizePath / destFilename
            # Call image magick as a binary to convert the images
            args = ['convert',str(sourcePath),str(destFilePath)]
            print('Running command: ' + ' '.join(args)+' ...')
            import subprocess
            subprocess.check_output(args=args)


def set_up_txt_subdirectory(inputPath=None,all_camera_parameters=None,destPath=None):
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
    assert destPath is not None, "destPath is a required argument!"
    assert (inputPath is None)!=(all_camera_parameters is None), 'Please pass either "images" or "inputPath" but not both!'

    txtPath = destPath / 'txt'
    if not txtPath.is_dir():
        txtPath.mkdir()

    if all_camera_parameters is None:
        from load_camera_info import load_all_camera_parameters
        all_camera_parameters = load_all_camera_parameters(inputPath, throw_error_if_radial_distortion=True)
    numCameras = len(all_camera_parameters)

    for i in range(numCameras):
        camera_parameters = all_camera_parameters[i]
        cameraMatrix = camera_parameters['camera_matrix']
        R = camera_parameters['R']
        T = camera_parameters['T']

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
                 level=2, # 0 is full resolution
                 csize=2, # cell size
                 threshold=0.5,
                 wsize=8, # window size
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
    ring topology and a configurable number of neighbors to be used for reconstruction.
    
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

def write_vis_file_sphere(numCameras, visFilePath=None, destPath=None, match_between_pairs=True):
    """ Generate a vis.dat file that pmvs2 expects for a camera with 
        cameras arranged in stereo pairs.
        Can also work for a ring as a special case, but there would be no matching between the pairs.
    Inputs:
    numCameras -- The number of cameras in the ring
    visFilePath -- Path of the vis file to be created. Pass this or:
    destPath -- Path of the directory where the vis.dat file will be created.
    """

    assert (visFilePath is None) != (destPath is None), 'Please pass one of visFilePath or destPath!'
    visFileName=Path('vis.dat')
    if visFilePath is None:
        visFilePath = destPath / visFileName

    assert numCameras%2==0, "write_vis_file_sphere expects an even number of cameras!"
    with visFilePath.open('w') as fd:
        fd.write('VISDATA\n')
        fd.write(str(numCameras)+'\n')
        if match_between_pairs:
            #above_pairings = ((1,8),(2,7),(3,12),(4,11),(5,10),(6,9)) # one based camera indeces
            above_pairings = ((0,7),(1,6),(2,11),(3,10),(4,9),(5,8)) # zero based camera indeces
            below_pairings = ( # Manually re-order by camera index
                    (6,1),
                    (7,0),
                    (8,5),
                    (9,4),
                    (10,3),
                    (11,2),
                    ) 
            for camera_index in range(numCameras):
                fd.write(str(camera_index)+' ')
                numNeighbors=3
                fd.write(str(numNeighbors)+' ')
                if camera_index < 6:
                    fd.write(str((camera_index-1)%6)+' ')
                    fd.write(str((camera_index+1)%6)+' ')
                    fd.write(str(above_pairings[camera_index][1])+' ')
                else:
                    fd.write(str((camera_index-6-1)%6+6)+' ')
                    fd.write(str((camera_index-6+1)%6+6)+' ')
                    fd.write(str(below_pairings[camera_index-6][1])+' ')
                fd.write('\n')
        else:
            for camera_index in range(numCameras):
                fd.write(str(camera_index)+' ')
                numNeighbors=1
                fd.write(str(numNeighbors)+' ')
                if camera_index%2==0:
                    fd.write(str(camera_index+1)+' ')
                else:
                    fd.write(str(camera_index-1)+' ')
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

    if all_camera_parameters is not None:
        numCameras = len(all_camera_parameters)
    else:
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

        This single function interface is convenient, but does all I/O every time.

    Arguments:
    imagesPath -- A directory full of source images
    destDir -- The destination directory of the ply file. (default current directory)
    destFile -- The destination name of the ply file. (default <name of the directory>.ply)
    options -- An instance of PMVS2Options
    workDirectory -- Existing directory where pmvs will work. (default generates a temp directory)
    runtimeFile -- The name of a file where info regarding the runtime will be stored.
    """

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


class PMVS2StereoMatcher():
    """ Wrapper class that calls PMVS2 on an array of cameras
        Usage: Re-instantiate each time the camera geometry changes with a new calibrationsPath.
            For each reconstruction, call either run_from_memory or run_from_disk depending on your use case.
    """
    def __init__(self,
            options=None,
            calibrationsPath=None,
            all_camera_parameters=None,
            work_directory=Path('pmvs2_work_directory'),
            ):

        assert (calibrationsPath is None) != (all_camera_parameters is None), "Please pass exactly one of all_camera_parameters or calibrationsPath!"
        if calibrationsPath is not None:
            from .load_camera_info import load_all_camera_parameters
            self.all_camera_parameters = load_all_camera_parameters(calibrationsPath)
        if all_camera_parameters is not None:
            self.all_camera_parameters = all_camera_parameters

        self.num_cameras = len(all_camera_parameters)
        if options is None:
            self.options = PMVS2Options(numCameras=self.num_cameras)
        else:
            self.options = options

        if not work_directory.is_dir():
            work_directory.mkdir()
        self.work_directory = work_directory

        # set_up_pmvs_tree
        set_up_txt_subdirectory(all_camera_parameters=all_camera_parameters,destPath=work_directory)

        # Generate the empty directory where pmvs puts its ply files
        modelsDir = work_directory / 'models'
        if not modelsDir.is_dir():
            modelsDir.mkdir()

        self.options.write_options_file(optionsDir=work_directory)

        # TODO I didn't do good bookkeeping yet in the database for the topology.
        # There should be a list of sensible matching topologies in all_camera_parameters,
        # and it shoud be added to the on-disk format.
        write_vis_file_sphere(self.num_cameras, destPath=work_directory, match_between_pairs=True)

    def run_from_memory(self, images, foreground_masks=None, dump_ply_files=False):
        """
        Run PMVS2 on images that are already in memory.
        Unfortunately, I still have to dump to disk, but at least I can hit the disk
        as little as possible.
        Inputs:
        images -- The ALREADY UNDISTORTED images
        foreground_masks -- unused, for API compatibility
        dump_ply_files -- also unused, for API compatiblity
        """
        # TODO Blow away previous images just to be safe?
        # TODO Blow away previous reconstruction just to be safe?

        # Dump out the new images into the already existing PMVS2 file tree
        set_up_visualize_subdirectory(images=images, destPath=self.work_directory)

        # Run PMVS2
        import subprocess
        from time import time
        args = [str(pmvs2Path), './', str('option.txt')] # Careful! That damn slash after the dot is CRITICAL
        print('Running command ', ' '.join(args))
        t1 = time()
        proc = subprocess.Popen(args=args, cwd=str(self.work_directory), stdout=subprocess.PIPE) # Python 3.4
        stdout, stderr = proc.communicate()
        t2 = time()
        dt = t2-t1
        returncode = proc.returncode
        print("pmvs2 output:")
        print(stdout.decode('utf8'))
        if returncode != 0:
            print("WARNING! pmvs2 returned a non-zero return value!")

        # Load the ply file from disk and return the xyz part
        modelsDir = self.work_directory / 'models'
        plyPath = modelsDir / Path('option.txt' + '.ply')
        assert plyPath.is_file(), 'PMVS2 did not generate a .ply file!'
        data, columnnames, columntypes = load_ply(plyPath, enableCaching=False)
        assert columnnames[0:3] == ['x', 'y', 'z']
        xyz = data[:, 0:3]
        return xyz,dt


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
    #run_pmvs(imagesPath, options=options) # to te
