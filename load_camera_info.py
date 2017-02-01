#!/usr/bin/env python3

import pathlib
from pathlib import Path
import numpy

_HalconDistortionParameters1 = ['Poly1', 'Poly2', 'Poly3', 'Poly4', 'Poly5']
_HalconDistortionParameters2 = ['Kappa']

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
    assert filePath.is_file()
    try:
        lines = filePath.open().readlines()
    except:
        print("File "+str(filePath)+" doesn't seem to be a readable text file! Try using HALCON 12 instead of 13 to generated the intrinsics!")
        raise
    lines = map(lambda line: line.strip(), lines)
    lines = filter(lambda line: line != '', lines)
    lines = filter(lambda line: line[0] != '#', lines)
    lines = map(lambda line: line.strip(), lines)
    lines = list(lines)

    # remove ParGroup header
    assert (lines[0].startswith('ParGroup'))
    currentLine = 2
    otherNames = ['Focus', 'Sx', 'Sy', 'Cx', 'Cy', 'ImageWidth', 'ImageHeight']
    expectedNames = _HalconDistortionParameters1 + _HalconDistortionParameters2 + otherNames
    d = {}
    while currentLine < len(lines):
        line = lines[currentLine]
        key = line.split(':')[0]
        assert key in expectedNames, 'Unhandled key found in intrinsics file!'
        value_string = line.split(':')[2].split(';')[0]
        if key in ('ImageWidth','ImageHeight'):
            float_value = float(value_string)
            from numpy import round, abs
            assert abs(round(float_value) - float_value)< .000001, key+' should be an integer!'
            value = int(round(float_value))
        else:
            value = float(value_string)
        currentLine += 3
        d[key] = value
    return d


def load_intrinsics(filePath):
    """ Load and convert the HALCON representation of the camera matrix
        into the representation closer to that used by open source
        programs.
        Input:
            filePath -- The name of the file to read
        Output:
            The 3x3 camera projection matrix K and distortion coefficients.
            x_pixel_homogeneous = K*x_world
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

    if 'Poly5' in d:
        k1 = d['Poly1']
        k2 = d['Poly2']
        k3 = d['Poly3']
        p1 = d['Poly4'] * .001
        p2 = d['Poly5'] * .001
        distCoeffs = (k1, k2, p1, p2, k3)
    elif 'Kappa' in d:
        distCoeffs = (d['Kappa'],)
    else:
        distCoeffs = ()

    return cameraMatrix, distCoeffs


def rodriguez_vector_to_SO3(a1,a2,a3, implementation='giplib'):
    """ Converts from an axis, angle rotation representation to a 3x3 matrix representation. 
        The angles are assumed to be in radians! """
    assert implementation in ['giplib','scipy']
    if implementation == 'giplib':
        # This implementation was ported over from giplib.
        from numpy import cos,sin,sqrt
        angle = sqrt(a1**2 + a2**2 + a3**2)
        if angle == 0:
            return numpy.eye(3)
        phi = angle
        rx = a1 / angle
        ry = a2 / angle
        rz = a3 / angle
        cosPhi = cos(phi)
        sinPhi = sin(phi)
        l_cosPhi = 1.0 - cosPhi
        rxRyl_cosPhi = rx * ry * l_cosPhi
        rxRzl_cosPhi = rx * rz * l_cosPhi
        return numpy.array([[ cosPhi+(rx**2)*l_cosPhi, rxRyl_cosPhi-rz*sinPhi, ry*sinPhi+rxRzl_cosPhi],
                            [rz*sinPhi+rxRyl_cosPhi, cosPhi+(ry**2)*l_cosPhi, ry*rz*l_cosPhi-rx*sinPhi],
                            [rxRzl_cosPhi-ry*sinPhi, rx*sinPhi+ry*rz*l_cosPhi, cosPhi+(rz**2)*l_cosPhi]])
    elif implementation == 'scipy':
        # A more straightforward way... not sure what is faster or more accurate.
        # Added this mainly to check the correctness of above port.
        import scipy
        a,b,c = a1,a2,a3
        skew = numpy.matrix([[0,-c,b],
                            [c,0,-a],
                            [-b,a,0]])
        return scipy.linalg.expm(skew)


def load_halcon_extrinsics_rodriguez(filePath):
    """ Load directly from HALCON's ascii format for camera extrinsics."""
    assert filePath.is_file()
    try:
        lines = filePath.open().readlines()
    except:
        print("File "+str(filePath)+" doesn't seem to be a readable text file containing extrinsics!")
        raise
    lines = map(lambda line: line.strip(), lines)
    lines = filter(lambda line: line != '', lines)
    lines = filter(lambda line: line[0] != '#', lines)
    lines = map(lambda line: line.strip(), lines)
    lines = list(lines)

    d = {}
    expected_keys = set(('f','r','t'))
    for line in lines:
        key = line[0]
        assert key in expected_keys, "Unrecognized character at start of line while parsing extrinsics!"
        expected_keys.remove(key)
        if line[0]=='f':
            assert line.split(' ')[1]=='0', 'Representation type not handled!'
            continue
        value = numpy.array(tuple(map(float,list(line.split(' '))[1:])))
        d[key] = value

    anglevector = d['r']*numpy.pi/180 # Supposedly the angles in the ascii files are in degrees... 
    R = rodriguez_vector_to_SO3(anglevector[0], anglevector[1], anglevector[2])
    t = d['t']
    return R,t


def load_halcon_extrinsics_homogeneous(filePath):
    """ HALCON is able to export camera extrinsics as a homogeneous matrix
        stored in an ascii text file. This export format is the easies to deal
        with.
        Input:
        filePath -- The path of the text file containing the homogeous matrix
        Output:
        The Rotation matrix and Translation vector associated with the camera.
        The matrices are for the transformation from the camera frame to
        world coordinate frame, i.e. x_world = R*x_camera + T
        T is in whatever units the camera calibration was done in (usually meters).
        If you need the transformation in the other direction you can do:
        R,T = R.T,numpy.dot(-R.T,T) # Invert the transform
        """
    strings = filePath.open().readlines()[0].strip().split(' ')
    assert len(strings)==12
    H = numpy.array(tuple(map(float,strings))).reshape((3,4))
    R = H[:,0:3]
    T = H[:,3]
    return R, T

def load_extrinsics(filePath):
    # Automatically detect the file format, so that this function
    # works with both types of ascii based HALCON extrinsics formats.
    try:
        lines = filePath.open().readlines()
    except:
        print("File "+str(filePath)+" doesn't seem to be a readable text file!")
        raise
    for line in lines:
        if 'Rotation angles [deg] or Rodriguez vector:' in line:
            #print('Rodriquez type HALCON extrinsics file detected!')
            return load_halcon_extrinsics_rodriguez(filePath)
    #print('Homogeneous type HALCON extrinsics file detected!')
    return load_halcon_extrinsics_homogeneous(filePath)
