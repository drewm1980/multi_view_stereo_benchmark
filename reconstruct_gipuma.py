#!/usr/bin/env python3

# Code for performing reconstruction using gipuma.
import subprocess
from pathlib import Path
from load_camera_info import load_intrinsics, load_extrinsics
import numpy

from reconstruct_pmvs2 import PMVS2Options

gipumaPath = Path('extern/gipuma/gipuma').resolve()

# Gipuma can, in principle, use pmvs2 directories, so we will try to re-use some code...
from reconstruct_pmvs2 import set_up_visualize_subdirectory, set_up_txt_subdirectory, write_vis_file_ring

def set_up_middlebury_tree(inputPath, destPath):
    """ Create a the input file and directory structure defined in the middlebury benchmark 
    Inputs: 
        inputPath -- full path to a directory containing undistorted images and HALCON camera calibration info.
        destPath -- full path of the directory in which the data and files will be placed.
    """
    import glob
    if not destPath.is_dir():
        destPath.mkdir()
    print('setting up middlebury tree in ', str(destPath), '...')
    numCameras = len(list((inputPath.glob("*.png"))))

    # Copy the images into destPath. To help keep things straight I will generate filenames
    # using "widgetSR" instead of "dinoSR" for sparse ring.
    pngFileNames = [] # Middleburry benchmark likes to stick names in files so we need to keep these.
    import shutil
    for i in range(numCameras):
        sourceFilename = "image_camera%02i.png" % (i + 1)
        destFilename = "widget%04i.png" % (i + 1)
        shutil.copy(
            str(inputPath / sourceFilename), str(destPath / destFilename))
        pngFileNames.append(destFilename)

    # Create the silhouette file
    Path(destPath/'widgetSR_good_silhouette_images.txt').touch()

    # Generate the par file
    """
    *_par.txt:  camera parameters.  There is one line for each image.  The format for each line is:
        "imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3"
        The projection matrix for that image is given by K*[R t]
        The image origin is top-left, with x increasing horizontally, y vertically
    *_ang.txt:  latitude, longitude angles for each image.  Not needed to compute scene->image mapping, but may be helpful for visualization.
    """
    parFileContents = ''
    for i in range(numCameras):
        # Load the intrinsics
        intrinsicsFilePath = inputPath / ('intrinsics_camera%02i.txt' % (i + 1))
        cameraMatrix, distCoffs = load_intrinsics(intrinsicsFilePath)
        # The images must already be radially undistorted
        assert(abs(distCoffs[0]) < .000000001)
        assert(abs(distCoffs[1]) < .000000001)
        assert(abs(distCoffs[2]) < .000000001)
        assert(abs(distCoffs[3]) < .000000001)
        assert(abs(distCoffs[4]) < .000000001)

        # Load the extrinsics
        extrinsicsFilePath = inputPath / ('extrinsics_camera%02i.txt' % (i + 1))
        R, T = load_extrinsics(extrinsicsFilePath)
        R,T = R.T,numpy.dot(-R.T,T) # Invert the transform

        # Load the intrinsics
        intrinsicsFilePath = inputPath / ('intrinsics_camera%02i.txt' % (i + 1))
        cameraMatrix, distCoffs = load_intrinsics(intrinsicsFilePath)
        K = cameraMatrix

        # TODO: Maybe I need to swap the meaning of x and y for middlebury?

        parFileContents += pngFileNames[i] + ' '
        assert cameraMatrix.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                parFileContents += str(float(K[i,j]))
                parFileContents += ' '
        for i in range(3):
            for j in range(3):
                parFileContents += str(float(R[i,j]))
                parFileContents += ' '
        for i in range(3):
            parFileContents += str(float(T[i]))
            parFileContents += ' '
        parFileContents = parFileContents[:-1]
        parFileContents += '\n'
    parFilePath = destPath / 'widgetSR_par.txt'
    #print('parFileContents: \n', parFileContents)
    parFilePath.open('w').write(parFileContents)

    # TODO: write the dinoSR_ang.txt file in case some algorithm needs it
    return pngFileNames

class GIPUMAOptions():
    """ This class represents the user-supplied options to GIPUMA.
        For argument descriptions see 
        https://github.com/kysucix/gipuma/wiki
        https://github.com/kysucix/gipuma
        """
    def __init__(self,
                input_type = ('pmvs','middlebury')[1],
                 camera_idx=None,
                 blocksize=None,
                 iterations=None,
                 min_angle=None,
                 max_angle=None,
                 max_views=None,
                 depth_min=None,
                 depth_max=None):
        self.input_type = input_type
        if self.input_type == 'pmvs':
            self.pmvs_folder = Path('.')
        if self.input_type == 'middlebury':
            self.krt_file = 'widgetSR_par.txt'
        #self.p_folder = p_folder

        self.camera_idx = camera_idx
        self.blocksize = blocksize
        self.iterations = iterations
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_views = max_views
        self.depth_min = depth_min
        self.depth_max = depth_max

    def to_parameter_string(self, camera_idx=None):

        s = ''

        # The required named parameters according to the wiki. 
        # According to the wiki these take a single dash and a space.
        if self.input_type == 'pmvs':
            s += '-pmvs_folder ' + str(self.pmvs_folder) # Careful! No leading space or the split on space later causes an empty parameter!
        if self.input_type == 'middlebury':
            s += '-krt_file ' + str(self.krt_file)

        # The optional named parameters according to the wiki.
        # According to the wiki they take double-dashes and an equals sign.
        assert camera_idx is not None or self.camera_idx is not None, 'option camera_idx must be set!'
        # the function parameter takes precedence if set.
        if camera_idx is None and self.camera_idx is not None:
            camera_idx = self.camera_idx
        s += ' --camera_idx=' + str(int(camera_idx))
        if self.blocksize is not None:
            s += ' --blocksize=' + str(int(self.blocksize))
        if self.iterations is not None:
            s += ' --iterations=' + str(int(self.iterations))
        if self.min_angle is not None:
            s += ' --min_angle=' + str(float(self.min_angle))
        if self.max_angle is not None:
            s += ' --max_angle=' + str(float(self.max_angle))
        if self.max_views is not None:
            s += ' --max_views=' + str(int(self.max_views))
        if self.depth_min is not None:
            s += ' --depth_min=' + str(float(self.depth_min))
        if self.depth_max is not None:
            s += ' --depth_max=' + str(float(self.depth_max))
        return s

def run_gipuma(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None):
    """ Run gipuma and fusibile on a directory full of images with HALCON style camera parameters

        The images must ALREADY be radially undistorted!

    Arguments:
    imagesPath -- A directory full of source images and HALCON style camera parameters
    destDir -- The destination directory of the ply file. (default current directory)
    destFile -- The destination name of the ply file. (default <name of the directory>.ply)
    options -- An instance of GIPUMAOptions or None
    workDirectory -- Existing directory where gipuma will work. (default generates a temp directory)
    runtimeFile -- The name of a file where info regarding the runtime will be stored.
    """
    import shutil
    import glob

    # By default, work in a temporary directory.
    # "with...as" ensures the temp directory is cleared even if there is an error below.
    if workDirectory is None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as workDirectory:
            run_gipuma(imagesPath=imagesPath,
                     destDir=destDir,
                     destFile=destFile,
                     options=options,
                     runtimeFile=runtimeFile,
                     workDirectory=Path(workDirectory))
        return
    if options is None:
        options = GIPUMAOptions(krt_file='widgetSR_par.txt')

    imagesPath = imagesPath.resolve()
    if options.input_type is 'pmvs':
        set_up_pmvs_tree(imagesPath, workDirectory, options=options)
    elif options.input_type is 'middlebury':
        pngFileNames = set_up_middlebury_tree(imagesPath, workDirectory)
    else:
        assert False, 'Unhandled input_type! Unable to detect if using pmvs or middlebury style input!'

    """ With gipuma and fusibile, it is apparently the user's responsibility to run gipuma multiple times and then run fusibile. This function automates that."""
    from time import time
    t1 = time()
    #pngFileNamesString = ' ' + ' '.join(pngFileNames) 
    #for camera_idx in range(numCameras): TODO change back
    for camera_idx in range(1):
        print('Runing gipuma for camera_idx = ', str(camera_idx), '...')
        flags = options.to_parameter_string(camera_idx).split(' ')

        if options.input_type=='middlebury':
            args = [str(gipumaPath)] + pngFileNames + flags
        else:
            args = [str(gipumaPath)] + flags
        print('Running command ', ' '.join(args), ' in directiory ', str(workDirectory) )
        result = subprocess.run(args=args, cwd=str(workDirectory), stdout=subprocess.PIPE)

    # Run fusibile to merge the depth maps.

    # TODO: There are a bunch of undocumented parameters, some of which seem to have been added just for the
    # middlebury benchmark, i.e. -remove_black_background
    # See extern/gipuma/scripts/dinoSparseRing.sh

    t2 = time()
    dt = t2-t1 # seconds. TODO: scrape more accurate timing from shell output
    print("gipuma output:")
    print(result.stdout.decode('utf8'))

    if result.returncode != 0:
        print("WARNING! GIPUMA returned a non-zero return value!")

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

    plyPath = modelsDir / Path(str(optionsFile) + '.ply')
    if plyPath.is_file():
        plyPath.rename(destPath)
    else:
        print('modelsDir: ' + str(modelsDir))
        print('plyPath: ' + str(plyPath))
        assert False, ".ply file wasn't generated!"


# Some hard-coded options, roughly slow to fast
optionsDict = {
            'gipuma_1': GIPUMAOptions(),
            }
optionNames = optionsDict.keys()
destFileNames = {optionName:optionName+'.ply' for optionName in optionNames}

if __name__=='__main__':
    print('Attempting to run a reconstruction using gipuma')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_gipuma')
    if not workDirectory.is_dir():
        workDirectory.mkdir()
    run_gipuma(imagesPath, workDirectory=workDirectory, options=optionsDict['gipuma_1'])
