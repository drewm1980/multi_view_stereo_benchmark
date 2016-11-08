#!/usr/bin/env python3

# Code for performing reconstruction using gipuma.

from pathlib import Path

gipumaPath = Path('extern/gipuma/gipuma').resolve()

# Gipuma can, in principle, use pmvs2 directories, so we will try to re-use some code...
from reconstruct_pmvs2 import set_up_visualize_subdirectory, set_up_txt_subdirectory, write_vis_file_ring

class GIPUMAOptions():
    """ This class represents most of the user-supplied options to GIPUMA.
        For argument descriptions see 
        https://github.com/kysucix/gipuma/wiki
        https://github.com/kysucix/gipuma
        """
    def __init__(self,
                 numCameras,
                 level=1, # 0 is full resolution
                 csize=2, # cell size
                 threshold=0.6,
                 wsize=7, # colors
                 minImageNum=2,
                 CPU=8,
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
            self.timages = [-1, 0, numCameras]
        else:
            self.timages = timages
        self.oimages = oimages

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
                if type(val) is list:
                    fd.write(key + ' ' + ' '.join(map(str, val)) + '\n')
                    continue



def run_gipuma_using_pmvs_input(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None, runtimeFile=None):
    """ Run gipuma on a directory full of images.

        The images must ALREADY be radially undistorted!

    Arguments:
    imagesPath -- A directory full of source images
    destDir -- The destination directory of the ply file. (default current directory)
    destFile -- The destination name of the ply file. (default <name of the directory>.ply)
    options -- An instance of GIPUMAOptions
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
            run_gipuma_using_pmvs_input(imagesPath=imagesPath,
                     destDir=destDir,
                     destFile=destFile,
                     options=options,
                     runtimeFile=runtimeFile,
                     workDirectory=Path(workDirectory))
        return

    imagesPath = imagesPath.resolve()
    set_up_visualize_subdirectory(imagesPath,workDirectory)
    set_up_txt_subdirectory(imagesPath,workDirectory)

    modelsDir = workDirectory / 'models'
    if not modelsDir.is_dir():
        modelsDir.mkdir()
    optionsFile='option.txt'

    numCameras = len(list(imagesPath.glob('*.png')))

    # Generate PMVS options file
    if options is None:
        options = GIPUMAOptions(numCameras=numCameras)
    options.write_options_file(optionsDir=workDirectory,
                               optionsFile=optionsFile)

    # Generate PMVS vis.dat file
    write_vis_file_ring(numCameras=numCameras,
                        numNeighbors=options.numNeighbors,
                        visFilePath=workDirectory / 'vis.dat')

    # Run GIPUMA
    import subprocess
    print("Calling gipuma...")
    from time import time
    args = [str(gipumaPath),'--pmvs_folder .', '--camera_idx=00000000', '--depth_min=.12', '--depth_max=0.2']
    print('Running command ', ' '.join(args))
    t1 = time()
    subprocess.check_output(args=args, cwd=str(workDirectory))
    t2 = time()
    dt = t2-t1 # seconds. TODO: scrape more accurate timing from PMVS shell output

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
            'gipuma_1': GIPUMAOptions(numCameras=12, level=2, csize=2, numNeighbors=1),
            }
optionNames = optionsDict.keys()
destFileNames = {optionName:optionName+'.ply' for optionName in optionNames}

if __name__=='__main__':
    print('Attempting to run a reconstruction using gipuma')
    imagesPath = Path('data/undistorted_images/2016_10_24__17_43_02')
    workDirectory=Path('working_directory_gipuma')
    #run_gipuma_using_pmvs_input(imagesPath, workDirectory=workDirectory)


