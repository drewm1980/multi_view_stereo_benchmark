#!/usr/bin/env python3

from pathlib import Path
import numpy


def set_up_visualize_subdirectory(inputPath,destPath):
    """
    Inputs:
    inputPath -- full path to a directory containing undistorted images
    destPath -- full path to a directory where the "visualize" subdir will be 
                created
    """
    import subprocess
    import glob
    visualizePath = destPath / 'visualize'
    visualizePath.mkdir()

    print('Setting up visualize subdirectory in ' + str(visualizePath)+'...')

    numCameras = len(list((inputPath.glob("*.png"))))
    for i in range(numCameras):
        sourceFilename = "image_camera%02i.png"%(i+1)
        destFilename = "%08i.ppm"%(i)
        sourcePath = inputPath / sourceFilename
        destPath = visualizePath / destFilename
        # Call image magick as a binary to convert the images
        args = ['convert',str(sourcePath),str(destPath)]
        print('Running command: ' + ' '.join(args)+' ...')
        subprocess.check_output(args=args)


def load_halcon_intrinsics(filePath):
    """ Load a halcon camera intrinsics file.
            i.e. the human-readable ASCII ones starting with \"ParGroup\"
        This function just does a 1:1 mapping of the (badly documented)
        file contents into python.
        Input:
            filePath -- The name of the file to read
        Output:
            A dictionary containing the focal length,
            radial distortion polynomial coefficients, etc...
            """
    lines = filePath.open().readlines()
    lines = map(lambda line: line.strip(), lines)
    lines = filter(lambda line: line != '', lines)
    lines = filter(lambda line: line[0] != '#', lines)
    lines = map(lambda line: line.strip(), lines)
    lines = list(lines)

    # remove ParGroup header
    assert (lines[0].startswith('ParGroup'))
    currentLine = 2
    expectedNames = ['Focus', 'Poly1', 'Poly2', 'Poly3', 'Poly4', 'Poly5',
                     'Sx', 'Sy', 'Cx', 'Cy', 'ImageWidth', 'ImageHeight']
    expectedNameIndex = 0
    d = {}
    while currentLine < len(lines) and expectedNameIndex < len(expectedNames):
        line = lines[currentLine]
        expectedName = expectedNames[expectedNameIndex]
        assert (line.startswith(expectedName))
        value_string = line.split(':')[2].split(';')[0]
        value = float(value_string)
        currentLine += 3
        expectedNameIndex += 1
        d[expectedName] = value
    return d


def load_intrinsics(filePath):
    """ Load and convert the HALCON representation of the camera matrix
        into the representation closer to that used by open source
        programs.
        Input:
            filePath -- The name of the file to read
        Output:
            The camera projection matrix and distortion coefficients
        """
    d = load_halcon_intrinsics(filePath)
    cameraMatrix = numpy.zeros([3, 3])

    fx = d['Focus'] / d['Sx']
    fy = d['Focus'] / d['Sy']
    cameraMatrix[0, 0] = fx
    cameraMatrix[1, 1] = fy

    cx = d['Cx']
    cy = d['Cy']
    cameraMatrix[0, 2] = cx
    cameraMatrix[1, 2] = cy
    cameraMatrix[2, 2] = 1.0

    k1 = d['Poly1']
    k2 = d['Poly2']
    k3 = d['Poly3']
    p1 = d['Poly4'] * .001
    p2 = d['Poly5'] * .001
    distCoffs = (k1, k2, p1, p2, k3)

    return cameraMatrix, distCoffs

def load_extrinsics(filePath):
    """ HALCON is able to export camera extrinsics as a homogeneous matrix
        stored in an ascii text file. This export format is the easies to deal
        with.
        Input:
        filePath -- The path of the text file containing the homogeous matrix
        Output:
        The Rotation matrix and Translation vector associated with the camera.
        """
    strings = filePath.open().readlines()[0].strip().split(' ')
    assert len(strings)==12
    H = numpy.array(tuple(map(float,strings))).reshape((3,4))
    R = H[:,0:3]
    T = H[:,3]
    return R, T

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
    txtPath.mkdir()

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

def run_pmvs(imagesPath, destDir=None, destFile=None, options=None, workDirectory=None):
    """ Run PMVS2 on a directory full of images.

        The images must ALREADY be radially undistorted!

    Arguments:
    imagesPath -- A directory full of source images
    destDir -- The destination directory of the ply file. (default current directory)
    destFile -- The destination name of the ply file. (default <name of the directory>.ply)
    options -- An instance of PMVS2Options
    workDirectory -- Existing directory where pmvs will work. (default generates a temp directory)
    """
    import shutil
    import glob

    # By default, work in a temporary directory.
    # "with...as" ensures the temp directory is cleared even if there is an error below.
    if workDirectory is None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as workDirectory:
            run_pmvs(imagesPath=imagesPath,
                     destDir=destDir,
                     destFile=destFile,
                     options=options,
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
        options = PMVS2Options(numCameras=numCameras)
    options.write_options_file(optionsDir=workDirectory,
                               optionsFile=optionsFile)

    # Generate PMVS vis.dat file
    write_vis_file_ring(numCameras=numCameras,
                        numNeighbors=options.numNeighbors,
                        visFilePath=workDirectory / 'vis.dat')

    # Run PMVS2
    import subprocess
    print("Calling pmvs2...")
    subprocess.check_output(args=['pmvs2', './', str(optionsFile)],
                            cwd=str(workDirectory))

    # Copy the file to the appropriate destination
    if destDir is None:
        destDir = Path.cwd()
    if destFile is None:
        destFile = 'reconstruction.ply'
    destPath = destDir / destFile

    plyPath = modelsDir / Path(str(optionsFile) + '.ply')
    if plyPath.is_file():
        plyPath.rename(destPath)
    else:
        print(".ply file wasn't generated!")
        print('modelsDir: ' + str(modelsDir))
        print('plyPath: ' + str(plyPath))
        assert False # Note, if this is hit, the tmp directory will already be removed!


# Some hard-coded options
optionNames = ['low', 'medium', 'high']
optionsDict = {'low': PMVS2Options(numCameras=12, level=2, csize=4, numNeighbors=1),
            'medium':PMVS2Options(numCameras=12, level=1, csize=4, numNeighbors=2),
            'high':PMVS2Options(numCameras=12, level=0, csize=4, numNeighbors=2)}
destFileNames = {optionName:optionName+'_quality.ply' for optionName in optionNames}

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

def do_reconstructions_for_the_benchmark(sourceDir=Path('undistorted'),
                                         destDir=Path('reconstructions')):
    assert sourceDir.is_dir()
    assert destDir.is_dir()
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
