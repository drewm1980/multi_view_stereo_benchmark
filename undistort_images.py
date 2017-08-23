#!/usr/bin/env python3

# Code for performing undistortion of images.
# Moved out of camera_models.py to resolve dependency issues.

import numpy
from numpy import sqrt
from numba import jit

import mvs
from mvs import lookup_monochrome as lookup_monochrome_python
lookup_monochrome = jit(lookup_monochrome_python)

from .camera_models import distort_division

@jit(cache=True)
#def undistort_image_slow(im, pixel_h, pixel_w, cx, cy, k1,k2,k3,p1,p2):
def undistort_image_halcon_division_no_lut(im, pixel_h, pixel_w, cx, cy, kappa):
    # Takes cx, cy in pixels
    # Takes pixel_h, pixel_w in m, like sx,sy in the HALCON .dat files.
    output_image = numpy.zeros_like(im)
    for vi in range(im.shape[0]):
        v = (numpy.float(vi) - cy) * pixel_h
        for ui in range(im.shape[1]):
            u = (numpy.float(ui) - cx) * pixel_w
            #k1,k2,k3,p1,p2 = (0.0,0.0,0.0,0.0,0.0) # For testing. Still broken with these set to zero.
            #u_tilde,v_tilde = distort_halcon_polynomial(u,v,k1,k2,k3,p1,p2)
            u_tilde, v_tilde = distort_division(u,v,kappa)
            
            #u_tilde,v_tilde = u,v # for debugging. Image gets through without distort_polynomial, so problem is in there.

            # Convert back to pixel indeces
            ui_tilde = u_tilde/pixel_w + cx
            vi_tilde = v_tilde/pixel_h + cy

            #ui_tilde,vi_tilde = ui,vi # for testing if lookup is at least working. lookup is working

            # Do image bounds check
            if ui_tilde < 0.0:
                continue
            if ui_tilde > im.shape[1]:
                continue
            if vi_tilde < 0.0:
                continue
            if vi_tilde > im.shape[0]:
                continue

            # Do bilinear interpolation based lookup
            intensity = lookup_monochrome(im, ui_tilde, vi_tilde)
            output_image[vi, ui] = intensity
    return output_image

def undistort_images(distorted_images, all_camera_parameters):
    """ Undistort several images. """
    # TODO try parallelizing? See background_subtraction for futures example.
    undistorted_images = []
    for i in range(len(distorted_images)):
        assert all_camera_parameters[i]['model']=='halcon_area_scan_division', 'Only halcon division model supported for undistortion, currently!'
        distorted_image = distorted_images[i]
        kappa = all_camera_parameters[i]['kappa']
        cx = all_camera_parameters[i]['cx']
        cy = all_camera_parameters[i]['cy']
        pixel_h = all_camera_parameters[i]['pixel_h']
        pixel_w = all_camera_parameters[i]['pixel_w']
        undistorted_image = undistort_image_halcon_division_no_lut(distorted_image, pixel_h, pixel_w, cx, cy, kappa)
        undistorted_images.append(undistorted_image)
    return undistorted_images
