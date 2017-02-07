#!/usr/bin/env python3

# Test code for functions that load halcon intrinsics and extrinsics
from load_camera_info import *

testDataPath = Path('data_for_unit_tests/extrinsics_for_testing')

def test_that_pose_types_are_consistent():
    num_cameras = 12
    for i in range(num_cameras):
        file_path_homogeneous = testDataPath / ('extrinsics_camera'+str(i+1).zfill(2)+'_homogeneous.txt')
        file_path_rodriguez = testDataPath / ('extrinsics_camera'+str(i+1).zfill(2)+'_rodriguez.txt') # not really rodriguez, just being consistent.
        R1,T1 = load_halcon_extrinsics_rodriguez(file_path_rodriguez)
        R2,T2 = load_halcon_extrinsics_homogeneous(file_path_homogeneous)
        eps = 1e-6
        assert (numpy.abs(R1-R2)<eps).all(), 'Rotation matrices do not match!'
        assert (numpy.abs(T1-T2)<eps).all(), 'Translation matrices do not match!'

if __name__=='__main__':
    test_that_pose_types_are_consistent()
