#!/usr/bin/env python3

import pathlib
from pathlib import Path

"""Code for comparing point clouds"""

cloud1Path = Path("./data/reconstructions/2016_10_24__17_43_17/reference.ply")
cloud2Path = Path("./data/reconstructions/2016_10_24__17_43_17/high_quality.ply")

from load_ply import load_ply

from ctypes import c_float as float32
from ctypes import c_int32 as int32
import numpy
from numpy.ctypeslib import ndpointer

import pandas

FloatPointer = ndpointer(dtype=numpy.float32, ndim=1, flags='ALIGNED', shape=(1,))

libcompare_clouds = numpy.ctypeslib.load_library('libcompare_clouds','.')
Cloud = ndpointer(dtype=numpy.float32,ndim=2,flags='CONTIGUOUS,ALIGNED')
'''This must MANUALLY be kept in sync with the PointCloudComparisonResult struct in compare_clouds.cpp! '''

PointCloudComparisonResultFields = ['numCloud1Points', 
                                    'numCloud2Points',
                                    'distanceThreshold',
                                    'numCloud1PointsNearCloud2',
                                    'numCloud2PointsNearCloud1']

PointCloudComparisonResultPointer = ndpointer(dtype=numpy.float32, ndim=1, flags='ALIGNED', shape=(len(PointCloudComparisonResultFields),))

def add_metadata(array):
    ''' Adds metatadata to the results of our point cloud comparison.'''
    a = pandas.DataFrame(data=array).T
    a.columns = PointCloudComparisonResultFields
    return a

# Wrap the PCL cloud comparison implementation if it was compiled into the .so file.
libcompare_clouds.compare_clouds_bruteforce.argtypes = [Cloud, Cloud, int32, int32, float32, PointCloudComparisonResultPointer]
USE_PCL = False
if 'compare_clouds_btree' in libcompare_clouds.__dict__.keys():
    libcompare_clouds.compare_clouds_btree.argtypes = [Cloud, Cloud, int32, int32, float32, float32, PointCloudComparisonResultPointer]
    USE_PCL = True
    def compare_clouds_btree(cloud1,
                       cloud2,
                       octreeResolution=.0075,
                       distanceThreshold=.002):
        """ octreeResolution and distanceThreshold are in the same units of distance as the point clouds. 
            octreeResolution has a BIG effect on run time; tune it if the benchmark is slow! """
        for cloud in (cloud1,cloud2):
            assert cloud.shape[1] == 3
        points1 = cloud1.shape[0]
        points2 = cloud2.shape[0]
        pointCloudComparisonResult = numpy.zeros((len(PointCloudComparisonResultFields),),dtype=numpy.float32)
        pointCloudComparisonResult[:] = numpy.nan
        libcompare_clouds.compare_clouds_btree(cloud1, cloud2, points1, points2,
                                               octreeResolution, distanceThreshold,
                                               pointCloudComparisonResult)
        return add_metadata(pointCloudComparisonResult)

def compare_clouds_bruteforce(cloud1,
                   cloud2,
                   distanceThreshold=.002):
    """ 
    This is just a sanity check to make sure the btree code isn't stupid slow.
        """
    for cloud in (cloud1,cloud2):
        assert cloud.shape[1] == 3
    points1 = cloud1.shape[0]
    points2 = cloud2.shape[0]
    pointCloudComparisonResult = numpy.zeros((len(PointCloudComparisonResultFields),),dtype=numpy.float32)
    pointCloudComparisonResult[:] = numpy.nan
    libcompare_clouds.compare_clouds_bruteforce(cloud1, cloud2, points1,
                                                points2, distanceThreshold,
                                                pointCloudComparisonResult)
    return add_metadata(pointCloudComparisonResult)

#compare_clouds = compare_clouds_btree # about 6 sec
compare_clouds = compare_clouds_bruteforce # about 6 sec

if __name__=='__main__':
    print('loading cloud 1...')
    cloud1PointData = load_ply(cloud1Path)[0][:,:3].astype(numpy.float32)
    print('loading cloud 2...')
    cloud2PointData = load_ply(cloud2Path)[0][:,:3].astype(numpy.float32)
    #print('Calling C++ compare_clouds code from python...')

    from time import time

    print('Running bruteforce point cloud comparison as a sanity check...')
    t1 = time()
    compare_clouds_bruteforce(cloud1PointData, cloud2PointData)
    #compare_clouds_btree(cloud1PointData, cloud2PointData)
    t2 = time()
    print(' time: ', t2-t1, ' sec')

    #for octreeResolution in (.001, .01, .1):
    #for octreeResolution in (.005, .01, .02):
    #for octreeResolution in (.005, .007, .01):
    #for octreeResolution in (.006, .007, .008):
    #for octreeResolution in (.0065, .007, .0075):
    #for octreeResolution in (.0075,):
    ##print('octreeResolution: ', octreeResolution, flush=True)
    #t1 = time()
    #compare_clouds_btree(cloud1PointData, cloud2PointData, octreeResolution=octreeResolution)
    #t2 = time()
    #print('octreeResolution: ', octreeResolution, end='')
    #print(' time: ', t2-t1, ' sec')
