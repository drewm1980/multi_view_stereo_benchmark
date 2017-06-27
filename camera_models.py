#!/usr/bin/env python3

# Python implementations of HALCON's radial distortion models.
# These are intended to work both element-wise for numpy arrays,
# and also for scalars.
# Transcribed from the documentation for the "calibrate_cameras" HALCON operator.
#
# I performed some manual common subexpresson elimination since
# numpy isn't a compiler and won't do CSE.

import numpy
from numpy import sqrt
from numba import jit

import mvs

halcon_model_types = ['area_scan_division', 'area_scan_polynomial']

def undistort_division(u_tilde, v_tilde, kappa):
    """ 
    From the HALCON Docs:
    
    The division model uses one parameter () to model the radial distortions.
    
    The following equations transform the distorted image plane coordinates into undistorted image plane coordinates if the division model is used """
    r_tilde_squared = u_tilde**2 + v_tilde**2
    scaling = 1.0 / (1.0 + kappa * r_tilde_squared)
    u = scaling * u_tilde
    v = scaling * v_tilde
    return u, v

def distort_division(u, v, kappa):
    """
    From the HALCON Docs:

    These equations can be inverted analytically, which leads to the following equations that transform undistorted coordinates into distorted coordinates:
    kappa = 0 means no distortion.
    """
    r_squared = u**2 + v**2
    temp = 1.0 - 4.0 * kappa * r_squared
    scaling = 2.0 / (1.0 + sqrt(temp))
    #print('scaling=',scaling)
    u_tilde = scaling * u
    v_tilde = scaling * v
    return u_tilde, v_tilde

@jit
def undistort_polynomial(u_tilde, v_tilde, k1, k2, k3, p1, p2):
    """
    From the HALCON Docs:
    The polynomial model uses three parameters () to model 
    the radial distortions and two parameters () to model 
    the decentering distortions.

    The following equations transform the distorted image 
    plane coordinates into undistorted image plane coordinates 
    if the polynomial model is used:

    These equations cannot be inverted analytically. Therefore, 
    distorted image plane coordinates must be calculated from 
    undistorted image plane coordinates numerically.

    k1=k2=k3=p1=p2=p3=0 means no distortion.
    """
    u_tilde_to_2 = u_tilde**2
    v_tilde_to_2 = v_tilde**2
    r_to_2 = u_tilde_to_2 + v_tilde_to_2
    r_to_4 = r_to_2**2
    r_to_6 = r_to_4 * r_to_2
    temp1 = k1 * r_to_2 + k2 * r_to_4 + k3 * r_to_6
    uv_tilde = u_tilde * v_tilde
    u = u_tilde + u_tilde * temp1 + p1 * (r_to_2 + 2 * u_tilde_to_2) + 2 * p2 * uv_tilde
    v = v_tilde = v_tilde * temp1 + 2 * p1 * uv_tilde + p2 * (r_to_2 + 2 * v_tilde_to_2)
    #print("Warning, I suspect the HALCON documentation is WRONG about the direction of this transformation.")
    return u,v

@jit
def distort_polynomial(u,v,k1,k2,k3,p1,p2):
    u_tilde,v_tilde = undistort_polynomial(u, v, k1, k2, k3, p1, p2)
    return u_tilde,v_tilde

from mvs import lookup_monochrome as lookup_monochrome_python
lookup_monochrome = jit(lookup_monochrome_python)

@jit
def undistort_image_slow(im, pixel_h, pixel_w, cx, cy, k1,k2,k3,p1,p2):
    # Takes cx, cy in pixels
    # Takes pixel_h, pixel_w in m, like sx,sy in the HALCON .dat files.
    output_image = numpy.zeros_like(im)
    for vi in range(im.shape[0]):
        v = (numpy.float(vi) - cy) * pixel_h
        for ui in range(im.shape[1]):
            u = (numpy.float(ui) - cx) * pixel_w
            u_tilde,v_tilde = distort_polynomial(u,v,k1,k2,k3,p1,p2)
            # Convert back to pixel indeces
            ui_tilde = u_tilde/pixel_w + cx
            vi_tilde = v_tilde/pixel_h + cy

            # Do bilinear interpolation based lookup
            intensity = lookup_monochrome(im, ui_tilde, vi_tilde)
            output_image[vi, ui] = intensity
    return output_image

def project_and_distort(x, y, z, f, sensor_h, sensor_w, pixel_h, pixel_w, cx,
                        cy, kappa=None):
    """ Project a 3D point into a sensor plane and
    simulate lens distortion to get (sub)pixel coordinates.
    This applies the camera's intrinsic / internal parameters.
    The caller is responsible for first applying the cameras extrinsic / external parameters
    to get the point's location in the camera frame.

    Note that the definition of the projection/distortion process is consistent with HALCON,
    but there exist other conventions, so be careful!. i.e. HALCON's radial distortion is applied
    to coordinates in the image plane (in m); I have seen other groups do radal distortion on the
    pixel coordinates.
    
    Inputs:
    x,y,z - real world coordinates in the camera frame. Scale doesn't matter.
    f - focal length of the lens in mm! This is to be consistent with giplib.
    sensor_h,sensor_w - height and width of the sensor in pixels
    pixel_h,pixel_w - height and width of a sensor pixel in mm! This is to be consistent with giplib.
    cx,cy - the center of the sensor optical axis in pixels
    kappa - the division model radial distortion parameter as defined by halcon.  """
    if kappa is None:
        kappa = 0.0
        print('Warning, kappa was not explicitly set!')
    f_meters = f * 0.001
    x_projected = x * f_meters / z
    y_projected = y * f_meters / z
    # z_projected = f_meters

    # Apply radial distortion in the image plane
    u = x_projected
    v = y_projected
    if kappa != 0.0 and kappa != -0.0:
        print('Non-zero Kappa! Applying radial distortion!')
        u_tilde, v_tilde = distort_division(u, v, kappa)
    else:
        u_tilde, v_tilde = u, v

    # Convert to pixel (sub) coordinates
    u_pixel = u_tilde / (pixel_w * .001) + cx
    v_pixel = v_tilde / (pixel_h * .001) + cy

    #assert False

    return u_pixel, v_pixel


# Check division model invertibility for a specific camera
def check_distortion_model_invertability(intrinsics):
    u_grid = numpy.arange(intrinsics['ImageWidth']).astype(numpy.float64)
    v_grid = numpy.arange(intrinsics['ImageHeight']).astype(numpy.float64)
    u, v= numpy.meshgrid(u_grid,v_grid,sparse=False)
    u -= intrinsics['Cx']
    v -= intrinsics['Cy']
    u *= intrinsics['Sx'] # in the HALCON .dat files it seems the values are in meters
    v *= intrinsics['Sy']
    kappa = intrinsics['Kappa']

    # Distort and undistort
    u_tilde,v_tilde = distort_division(u,v,kappa)
    from numpy import all, abs
    assert not all(abs(u_tilde-u)<intrinsics['Sx']), 'Warning: Distortion is at most sub-pixel! Probably a bug!'
    assert not all(abs(v_tilde-v)<intrinsics['Sy']), 'Warning: Distortion is at most sub-pixel! Probably a bug!'
    u2,v2 = undistort_division(u_tilde,v_tilde,kappa)
    eps = .001*intrinsics['Sx'] # a thousandth of a pixel
    assert all(abs(u-u2)<eps) and all(abs(v-v2)<eps), 'Camera intrinsics are not invertible on the image domain!'

    # Undistort then Distort
    u_tilde,v_tilde = undistort_division(u,v,kappa)
    assert not all(abs(u_tilde-u)<intrinsics['Sx']), 'Warning: Distortion is at most sub-pixel! Probably a bug!'
    assert not all(abs(v_tilde-v)<intrinsics['Sy']), 'Warning: Distortion is at most sub-pixel! Probably a bug!'
    u2,v2 = distort_division(u_tilde,v_tilde,kappa)
    assert all(abs(u-u2)<eps) and all(abs(v-v2)<eps), 'Camera intrinsics are not invertible on the image domain!'

if __name__=='__main__':
    test_division_model_invertability(1)
    test_division_model_invertability(20)
