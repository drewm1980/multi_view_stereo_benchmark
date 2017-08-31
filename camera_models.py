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

@jit
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
def undistort_halcon_polynomial(u_tilde, v_tilde, k1, k2, k3, p1, p2):
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

    k1=k2=k3=p1=p2=0 means no distortion.
    """
    u_tilde_to_2 = u_tilde**2
    v_tilde_to_2 = v_tilde**2
    r_to_2 = u_tilde_to_2 + v_tilde_to_2
    r_to_4 = r_to_2**2
    r_to_6 = r_to_4 * r_to_2
    temp1 =  k1 * r_to_2
    temp1 += k2 * r_to_4
    temp1 += k3 * r_to_6
    u = u_tilde + u_tilde * temp1  # the radial part
    v = v_tilde + v_tilde * temp1
    uv_tilde = u_tilde * v_tilde
    u += p1 * (r_to_2 + 2 * u_tilde_to_2) + 2 * p2 * uv_tilde # The tilt part
    v += 2 * p1 * uv_tilde + p2 * (r_to_2 + 2 * v_tilde_to_2)
    return u,v

@jit
def distort_halcon_polynomial(u,v,k1,k2,k3,p1,p2):
    #u_tilde,v_tilde = undistort_polynomial(u, v, k1, k2, k3, p1, p2)
    assert False, "Not implemented yet. Requires iterative solution"
    return u_tilde,v_tilde

def project_and_distort_simple(point_3d, camera_parameters):
    """ Simplified api that takes a 3d point, applies extrinsics, intrinsics,
    and radial distortion to return a 2d image point corresponding to a 3d point.
    This implementation is not for speed critical use; see project_and_distort. """
    R = camera_parameters['R']
    T = camera_parameters['T']
    kappa = camera_parameters['kappa']
    f_mm = camera_parameters['f'] * 1000.0 # Because project_and_distort takes mm
    image_height = camera_parameters['image_height']
    image_width = camera_parameters['image_width']
    pixel_h_mm = camera_parameters['pixel_h'] * 1000.0
    pixel_w_mm = camera_parameters['pixel_w'] * 1000.0
    cx = camera_parameters['cx']
    cy = camera_parameters['cy']
    x,y,z = numpy.dot(R,point_3d) + T
    assert z*1000.0 > f_mm, 'A voxel is behind the image plane of a camera!'
    u, v = project_and_distort(x, y, z, f_mm, image_height, image_width,
                               pixel_h_mm, pixel_w_mm, cx, cy, kappa)
    return u,v

@jit
def project_and_distort(x, y, z, f_mm, sensor_h, sensor_w, pixel_h_mm, pixel_w_mm, cx,
                        cy, kappa):
    """ Project a 3D point into a sensor plane and
    simulate lens distortion to get (sub)pixel coordinates.
    This applies the camera's intrinsic / internal parameters.
    The caller is responsible for first applying the cameras extrinsic / external parameters
    to get the point's location in the camera frame.

    Note that the definition of the projection/distortion process is consistent with HALCON,
    but there exist other conventions, so be careful!. i.e. HALCON's radial distortion is applied
    to coordinates in the image plane (in m); I have seen other groups do radial distortion on the
    pixel coordinates.

    Inputs:
    x,y,z - real world coordinates in the camera frame. Scale doesn't matter.
    f_mm - focal length of the lens in mm! This is to be consistent with giplib.
    sensor_h,sensor_w - height and width of the sensor in pixels
    pixel_h,pixel_w - height and width of a sensor pixel in mm! This is to be consistent with giplib.
    cx,cy - the center of the sensor optical axis in pixels
    kappa - the division model radial distortion parameter as defined by halcon.  """
    f_meters = f_mm * 0.001
    x_projected = x * f_meters / z
    y_projected = y * f_meters / z
    # z_projected = f_meters

    # Apply radial distortion in the image plane
    u = x_projected
    v = y_projected
    #if kappa != 0.0 and kappa != -0.0:
    #print('Non-zero Kappa! Applying radial distortion!')
    #u_tilde, v_tilde = distort_division(u, v, kappa)
    #else:
    #u_tilde, v_tilde = u, v
    u_tilde, v_tilde = distort_division(u, v, kappa)

    # Convert to pixel (sub) coordinates
    u_pixel = u_tilde / (pixel_w_mm * .001) + cx
    v_pixel = v_tilde / (pixel_h_mm * .001) + cy

    #assert False

    return u_pixel, v_pixel

def triangulate(camera_point_tuples):
    """ Takes (camera parameters, 2d point) tuples and computes the closest
        3d point to the viewing rays. 
        This is solves the inverse problem of project_and_distort_simple."""
    points_on_the_lines = []
    direction_vectors = []
    assert(len(camera_point_tuples)) >= 2, 'There are not enough points to triangulate!'
    for camera_parameters,(u_pixel,v_pixel) in camera_point_tuples:
        # Extract two points per ray in world coordinates.
        R = camera_parameters['R'] # R,T, map world coorinates into camera coordinates
        T = camera_parameters['T']
        R_,T_ = R.T,numpy.dot(-R.T,T) # map camera coordinates into world coordinates
        kappa = camera_parameters['kappa']
        f_m = camera_parameters['f'] # Because project_and_distort takes mm
        image_height = camera_parameters['image_height']
        image_width = camera_parameters['image_width']
        pixel_h_m = camera_parameters['pixel_h']
        pixel_w_m = camera_parameters['pixel_w']
        cx = camera_parameters['cx']
        cy = camera_parameters['cy']
        u_tilde = (u_pixel - cx) * pixel_w_m
        v_tilde = (v_pixel - cy) * pixel_h_m
        u,v = undistort_division(u_tilde, v_tilde, kappa)
        #point1_camera = numpy.array([0,0,0])
        point1_world = camera_center = T_ # in world coordinates since R_*[[0],[0],[0]] + T_ = T_
        point2_camera = numpy.array((u,v,f_m))
        point2_world = numpy.dot(R_,point2_camera)+T_
        direction_world = point2_world - point1_world
        points_on_the_lines.append(point1_world)
        direction_vectors.append(direction_world)
    from plaroma3d.camera_geometry import point_closest_to_several_lines
    point_3d_world = point_closest_to_several_lines(points=points_on_the_lines,
                                              directions=direction_vectors)
    point_to_line_distances = []
    for p,d in zip(points_on_the_lines, direction_vectors):
        n = d / numpy.linalg.norm(d) # normalized direction
        point_3d_p = point_3d_world - p # center on a point on the line
        # To compute distance, consider right triangle formed by the projection...
        distance = numpy.sqrt(numpy.dot(point_3d_p,point_3d_p) - (numpy.dot(point_3d_p,n))**2) 
        point_to_line_distances.append(distance)
        
    return point_3d_world, point_to_line_distances


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
