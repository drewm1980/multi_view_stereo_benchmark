#!/usr/bin/env python3

# Code for background modeling, subtraction, etc...

import numpy


def histogram_to_median_python(hist, expected_median=None):
    assert len(hist)==256, "histogram_to_median_python only works on arrays of length 256"
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
    
    return median
        #print('i',i,'sum_left',sum_left, 'sum_right',sum_right)
    # Followed a sequence of zeros to the end.
    #assert False

try:
    from numpy.ctypeslib import ndpointer
    from ctypes import c_float as float32
    from ctypes import c_int32 as int32
    libbackground_subtraction = numpy.ctypeslib.load_library('libbackground_subtraction','.')
    image_type = ndpointer(dtype=numpy.uint8,ndim=2,flags='CONTIGUOUS,ALIGNED')
    histogram_image_type = ndpointer(dtype=numpy.uint8,ndim=3,flags='CONTIGUOUS,ALIGNED')
    libbackground_subtraction.histogram_to_median.argtypes = (ndpointer(dtype=numpy.uint8,ndim=1,flags='CONTIGUOUS,ALIGNED'),)
    libbackground_subtraction.histogram_to_median.restype = numpy.uint8
    histogram_to_median_c = libbackground_subtraction.histogram_to_median
    histogram_to_median = histogram_to_median_c

    libbackground_subtraction.update_histogram_image.argtypes = (image_type, histogram_image_type, int32)
    def update_histogram_image(image, histogram_image):
        '''Take a histogram image stored as a h x w x 256 uint8 numpy array,
        and increment the values in-place with the values in the passed image.
        I couldn't find a way to do this efficiently with numpy indexing,
        thus the trivial C extension.'''
        rows,columns,bins = histogram_image.shape
        pixels = rows*columns
        assert bins==256
        libbackground_subtraction.update_histogram_image(image, histogram_image, pixels)

    libbackground_subtraction.median_of_histogram_image.argtypes = (histogram_image_type, image_type, int32)
    def median_of_histogram_image(histogram_image):
        rows,columns,bins = histogram_image.shape
        pixels = rows*columns
        assert bins==256
        median_image = numpy.empty((rows,columns),dtype=numpy.uint8)
        libbackground_subtraction.median_of_histogram_image(histogram_image, median_image, pixels)
        return median_image

    def pixelwise_median(images):
        ''' Compute the pixelwise median of a set of images.
            input is an iterable of images. Danger of saturation if run on more than 256 images!'''
        histogram_image = None
        expected_shape = None
        for image in images:
            assert image.dtype==numpy.uint8
            assert len(image.shape)==2, "Only monochrome images are currently supported!"
            # Note: trivial extension to color is possible by doing each channel separately, but
            #       this wouldn't always a good robust estimator within a 3D colorspace, so color abberations
            #       would probably occur at some pixels. 
            h,w = image.shape
            if not expected_shape:
                expected_shape = image.shape
            assert image.shape == expected_shape, 'Images must all be the same size!'
            assert image.dtype == numpy.uint8, 'Images must be type uint8!'
            if not histogram_image:
                histogram_image = numpy.zeros((h,w,256),dtype=numpy.uint8)
            update_histogram_image(image,histogram_image)
        return median_of_histogram_image(histogram_image)
            
except:
    print('Could not load C extension for histogram_to_median. Will default to python implementation!')
    histogram_to_median = histogram_to_median_python


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
        # Check edge cases at the end of a big array
        (255*(0,)+(1,),255),
        (255*(1,)+(2,),128),
        # Same but for a less special sized array
        (55*(0,)+(1,),55),
        (55*(1,)+(2,),28),
        )

end_test = (0*numpy.ones(256),127),
def test_histogram_to_median():
    for hist,expected_median in histogram_to_median_test_cases:
        hist = numpy.array(hist)
        hist_padded = numpy.zeros(256,dtype=numpy.uint8)
        hist_padded[0:len(hist)] = hist
        #print('hist=',hist_padded)
        #histogram_to_median(hist_padded, expected_median)
        assert histogram_to_median(hist_padded) == expected_median
    print('test_histogram_to_median: PASSED!')

def test_histogram_to_median_python_and_c_equivalence():
    for hist,expected_median in histogram_to_median_test_cases:
        hist = numpy.array(hist)
        hist_padded = numpy.zeros(256,dtype=numpy.uint8)
        hist_padded[0:len(hist)] = hist
        #print('hist=',hist_padded)
        median_c = histogram_to_median_c(hist_padded, expected_median)
        median_python = histogram_to_median_python(hist_padded, expected_median)
        assert median_c == expected_median
        assert median_c == median_python
    print('test_histogram_to_median_python_and_c_equivalence: PASSED!')

def test_pixelwise_median():
    h = 10
    w = 10
    noise = numpy.random.randint(low=0,high=10,size=(h,w),dtype=numpy.uint8)
    signal = numpy.random.randint(low=0,high=255,size=(h,w),dtype=numpy.uint8)
    images = 2*(signal,)+(signal+noise,) # One noisy image shouldn't throw off the median
    median_image = pixelwise_median(images)
    assert numpy.all(median_image==signal), 'Median failed to recover noiseless image in test_pixelwise_median!'
    print('test_pixelwise_median: PASSED')

if __name__=='__main__':
    test_histogram_to_median()
    test_histogram_to_median_python_and_c_equivalence()
    test_pixelwise_median()
