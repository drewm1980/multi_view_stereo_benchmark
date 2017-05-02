#!/usr/bin/env python3

# Code for background modeling, subtraction, etc...

from ctypes import c_float as float32
from ctypes import c_int32 as int32
import numpy
from numpy.ctypeslib import ndpointer

#libbackground_subtraction = numpy.ctypeslib.load_library('libbackground_subtraction','.')
#image_type = ndpointer(dtype=numpy.uint8,ndim=2,flags='CONTIGUOUS,ALIGNED')
#histogram_type = ndpointer(dtype=numpy.uint8,ndim=3,flags='CONTIGUOUS,ALIGNED')
#libbackground_subtraction.argtypes


def histogram_to_median(hist, expected_median=None):
    samples = numpy.sum(hist)
    
    # Explicitly handle edge cases
    if(hist[0]*2>samples): 
        return 0;
    if(hist[255]*2>samples):
        return 255;

    # Handle the middle cases
    sum_left = 0;
    sum_right = samples - hist[0];
    lowest_difference = sum_right;
    plateau_start=0
    plateau_end=0
    for i in range(256):
        difference = abs(sum_right - sum_left);
        print('sum_left=',sum_left,' sum_right=',sum_right,' difference = ',difference)
        if difference < lowest_difference:
            lowest_difference = difference
            plateau_start = i
            plateau_end = i
        if difference == lowest_difference:
            plateau_end = i
        if (difference > lowest_difference):
            plateau_end = i-1
            break
        if i<255:
            sum_left += hist[i];
            sum_right -= hist[i+1];

    median = int(numpy.floor((plateau_end+plateau_start)/2)); # Implements averaging
    if expected_median is not None:
        if expected_median != median:
            print('plateau_start=',plateau_start)
            print('plateau_end=',plateau_end)
            print('median=',median)
            print('expected_median=',expected_median)
            assert False
    
    return 
        #print('i',i,'sum_left',sum_left, 'sum_right',sum_right)
    # Followed a sequence of zeros to the end.
    #assert False

# Tuples containing hist as a tuple, and expected median. If smaller than 256 entries, will be zero padded.
histogram_to_median_test_cases = (
        ((1,),0),
        ((0,1,),1),
        ((0,0,1,),2),
        ((0,1,0),1),
        ((2,1),0),
        ((1,2),1),
        ((3,1),0),
        ((1,3),1),
        # hills
        ((1,2,1),1),
        ((1,2,3,2,1),2),
        # plateaus
        ((1,1),0),
        ((1,1,1),1),
        ((0,1,1,0),1),
        (0*numpy.ones(256),127),
        (1*numpy.ones(256),127),
        (2*numpy.ones(256),127),
        (3*numpy.ones(256),127),
        (254*numpy.ones(256),127),
        (255*numpy.ones(256),127),
        # gaps
        ((1,0,1),1),
        ((1,0,0,1),1),
        ((1,0,0,0,1),2),
        ((1,0,0,0,0,1),2),
        ((1,0,0,0,0,0,1),3),
        ((2,0,2),1),
        ((2,0,0,2),1),
        ((2,0,0,0,2),2),
        ((2,0,0,0,0,2),2),
        ((2,0,0,0,0,0,2),3),
        ((0,1,0,1,0,1,0),3),
        )

def test_histogram_to_median():
    for hist,expected_median in histogram_to_median_test_cases:
        hist = numpy.array(hist)
        hist_padded = numpy.zeros(256)
        hist_padded[0:len(hist)] = hist
        #assert histogram_to_median(hist_padded) == expected_median
        print('hist=',hist_padded)
        histogram_to_median(hist_padded, expected_median)

def update_histogram(image, histogram):
    # Take a histogram image stored as a h x w x 256 uint8 numpy array,
    # and increment the values in-place with the values in the passed image.
    # I couldn't find a way to do this efficiently with numpy indexing,
    # thus the trivial C extension.
    pass

if __name__=='__main__':
    test_histogram_to_median()
