#!/usr/bin/env python3

from pathlib import Path

"""Code for comparing point clouds"""

cloud1Path = Path("./data/reconstructions/2016_10_24__17_43_17/reference.ply")
cloud2Path = Path("./data/reconstructions/2016_10_24__17_43_17/high_quality.ply")

from load_ply import load_ply

from ctypes import c_float as float32
from ctypes import c_int32 as int32
import numpy
from numpy.ctypeslib import ndpointer

FloatPointer = ndpointer(dtype=numpy.float32, ndim=1, flags='ALIGNED', shape=(1,))

libcompare_clouds = numpy.ctypeslib.load_library('libcompare_clouds','.')
cloud_type = ndpointer(dtype=numpy.float32,ndim=2,flags='CONTIGUOUS,ALIGNED')
libcompare_clouds.compare_clouds.argtypes = [cloud_type, cloud_type, int32, int32]

def compare_clouds(cloud1, cloud2):
    for cloud in (cloud1,cloud2):
        assert cloud.shape[1] == 3
    points1 = cloud1.shape[0]
    points2 = cloud2.shape[0]
    return libcompare_clouds.compare_clouds(cloud1, cloud2, points1, points2)


if __name__=='__main__':
    cloud1PointData = load_ply(cloud1Path)[0][:,:3].astype(numpy.float32)
    cloud2PointData = load_ply(cloud2Path)[0][:,:3].astype(numpy.float32)
    compare_clouds(cloud1PointData, cloud2PointData)

