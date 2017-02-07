#!/usr/bin/env bash

# Code for testing camera models

from camera_models import *
import numpy
from numpy import all, abs

def test_division_model_invertability(tests=20, kappa=0):
    """ Test if the division actually inverts for some random u,v,kappa"""
    u_tilde = numpy.random.randn(tests)
    v_tilde = numpy.random.randn(tests)
    u,v = undistort_division(u_tilde,v_tilde,kappa)
    u_tilde2,v_tilde2 = distort_division(u,v,kappa)
    eps = .000001
    assert all(abs(u_tilde-u_tilde2)<eps)
    assert all(abs(v_tilde-v_tilde2)<eps)
    
if __name__=='__main__':
    #check_distortion_model_invertability(intrinsics)
    test_division_model_invertability(tests=20, kappa=0):



