#!/usr/bin/env python3
from load_ply import *
import numpy

def test_save_ply_using_library():
    xyz = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    save_ply_using_library(xyz,'foo.ply')
def test_save_ply_file():
    xyz = numpy.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    save_ply_file(xyz=xyz,filename='foo.ply')

if __name__=='__main__':
    test_save_ply_file()

