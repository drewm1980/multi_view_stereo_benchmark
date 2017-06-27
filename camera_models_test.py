#!/usr/bin/env bash

# Code for testing camera models

from camera_models import *
import numpy
from numpy import all, abs
from pathlib import Path

import mvs # Must be on the path somewhere

def test_division_model_invertability(tests=20, kappa=0):
    """ Test if the division actually inverts for some random u,v,kappa"""
    u_tilde = numpy.random.randn(tests)
    v_tilde = numpy.random.randn(tests)
    u,v = undistort_division(u_tilde,v_tilde,kappa)
    u_tilde2,v_tilde2 = distort_division(u,v,kappa)
    eps = .000001
    assert all(abs(u_tilde-u_tilde2)<eps)
    assert all(abs(v_tilde-v_tilde2)<eps)

def test_undistortion():
    """ Load an image and just look at it.
        Basically just a no-crash test. """
    testImage = Path('./data_for_unit_tests/image_with_barrel_distortion.png')
    from mvs import load_image_preserving_type
    im = load_image_preserving_type(testImage)
    from load_camera_info import load_halcon_intrinsics
    distorted_intrinsics = load_halcon_intrinsics(Path('data_for_unit_tests/intrinsics_polynomial01.dat'))
    #undistorted_intrinsics = load_halcon_intrinsics(Path('data_for_unit_tests/intrinsics_undistorted.txt'))
    pixel_h = distorted_intrinsics['Sy']
    pixel_w = distorted_intrinsics['Sx']
    cx = distorted_intrinsics['Cx']
    cy = distorted_intrinsics['Cy']
    k1 = distorted_intrinsics['Poly1']
    k2 = distorted_intrinsics['Poly2']
    k3 = distorted_intrinsics['Poly3']
    p1 = distorted_intrinsics['Poly4'] * .001
    p2 = distorted_intrinsics['Poly5'] * .001
    #k1,k2,k3,p1,p2 = (0,0,0,0,0) # For testing
    from time import time
    im_undistorted = undistort_image_slow(im, pixel_h, pixel_w, cx, cy, k1,k2,k3,p1,p2) # warmup
    t1 = time()
    im_undistorted = undistort_image_slow(im, pixel_h, pixel_w, cx, cy, k1,k2,k3,p1,p2) # 36 ms
    t2 = time()
    print("Undistorting the image without any lookup tables took ",t2-t1,"seconds")
    from skimage.io import imsave
    imsave('undistorted.png',im_undistorted)
    return im_undistorted
    
    

if __name__=='__main__':
    #check_distortion_model_invertability(intrinsics)
    #test_division_model_invertability(tests=20, kappa=0)
    im_undistorted = test_undistortion()



