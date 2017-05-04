#!/usr/bin/env python3

# Code for background modeling, subtraction, etc...

import numpy


def histogram_to_median_python(hist, expected_median=None):
    print('histogram=',hist)
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

        # For debug:
        if plateau_start>0:
            h1 = hist[plateau_start-1]
        else:
            h1 = hist[plateau_start]
        if plateau_end<255:
            h2 = hist[plateau_end+1]
        else:
            h2 = hist[plateau_end]
        print('plateau=',(plateau_start, plateau_end, h1, h2))

    if plateau_start == plateau_end:
        median = plateau_start
    elif h1==h2:
        median = int(numpy.floor((plateau_end+plateau_start)/2)); # Implements averaging
    elif h1 < h2:
        median = plateau_end
    else:
        median = plateau_start

    print('median=',median)
    print('expected_median=',expected_median)

    if expected_median is not None:
        if expected_median != median:
            #print('plateau_start=',plateau_start)
            #print('plateau_end=',plateau_end)
            print('median=',median)
            print('expected_median=',expected_median)
            assert False
    
    return median
        #print('i',i,'sum_left',sum_left, 'sum_right',sum_right)
    # Followed a sequence of zeros to the end.
    #assert False

def pixelwise_median_sort_based_python(images):
    # Implementaton of pixelwise median based on numpy's "partition", comparable to C++'s nth_element.
    # the actual underlying algorithm is introselect
    print('Stacking the images up in a big array...')
    big_array = numpy.dstack(images)
    num_images = big_array.shape[2]
    # A couple trivial cases
    if len(images)==1:
        return images[0]
    if len(images)==2:
        return (images[0].astype(numpy.int) + images[1].astype(numpy.int))/2
    if num_images%2==1:
        # Case of ODD number of images is easier
        kth=(num_images-1)/2
        print('Performing partition for the odd case...')
        big_array.partition(kth, axis=2)
        print('Extracting the median into a dense array...')
        return big_array[:,:,kth].copy()
    else:
        # Case of EVEN number of images requires some averaging.
        # numpy's partition is able to get the middle TWO positions
        # into their sorted order.
        middle_two=(num_images/2-1,num_images/2)
        print('Performing partition for the even case...')
        big_array.partition(kth=middle_two,axis=2)
        print('Doing averaging and extracting the median into a dense array...')
        return big_array[:,:,int(middle_two[0]):int(middle_two[0]+2)].mean(axis=2).astype(images[0].dtype) # flooring conversion if images are integer pixel type.

def pixelwise_median_numpy(images):
    print('Stacking the images up in a big array...') # This is the slow part!
    big_array = numpy.dstack(images)
    median_image = numpy.empty_like(images[0])
    print("Calling numpy's in-place median function...")
    numpy.median(big_array, axis=2, out=median_image, overwrite_input=True)
    return median_image


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

    def pixelwise_median_histogram_based_c(images):
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
            if expected_shape is None:
                expected_shape = image.shape
            assert image.shape == expected_shape, 'Images must all be the same size!'
            assert image.dtype == numpy.uint8, 'Images must be type uint8!'
            if histogram_image is None:
                histogram_image = numpy.zeros((h,w,256),dtype=numpy.uint8)
            update_histogram_image(image,histogram_image)
        print(histogram_image[0,0,:])
        #assert False
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
        ((0,2,0,1),1), # Reproduce a bug in plateau logic
        (4*(0,)+(2,)+16*(0,)+(1,),4), # Reproduce a bug in plateau logic
        (103*(0,)+(128,)+5*(0,)+(127,),104), # Reproduce a bug in plateau logic
        )

def test_histogram_to_median_python():
    for hist,expected_median in histogram_to_median_test_cases:
        hist = numpy.array(hist)
        hist_padded = numpy.zeros(256,dtype=numpy.uint8)
        hist_padded[0:len(hist)] = hist
        histogram_to_median_python(hist_padded, expected_median) # Better for debugging
        assert histogram_to_median_python(hist_padded) == expected_median
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
    h = 1024
    w = 1024
    numpy.random.seed(10)
    noise = numpy.random.randint(low=0,high=10,size=(h,w),dtype=numpy.uint8)
    signal = numpy.random.randint(low=0,high=255,size=(h,w),dtype=numpy.uint8)
    images = 128*(signal,)+127*(signal+noise,) # One noisy image shouldn't throw off the median... OOPs wraparound!
    from time import time
    t1 = time()
    #median_image = pixelwise_median_histogram_based_c(images) # Broken C implementation
    #median_image = pixelwise_median_sort_based_python(images) 
    median_image = pixelwise_median_numpy(images) 
    t2 = time()
    print('Median computation took', t2-t1, ' seconds.')
    if not numpy.all(median_image==signal):
        print('Median failed to recover noiseless image in test_pixelwise_median!')
        failed_pixels = numpy.argwhere(median_image != signal)
        print('Failed cases:')
        for pixelx,pixely in failed_pixels:
            single_pixel_image = tuple([image[pixelx,pixely] for image in images])
            signal_pixel = signal[pixelx,pixely]
            noise_pixel = noise[pixelx,pixely]
            proposed_test_case = (single_pixel_image, signal_pixel)
            print('proposed_test_case:',proposed_test_case)

    print('test_pixelwise_median: PASSED')

if __name__=='__main__':
    #test_histogram_to_median_python()
    #test_histogram_to_median_python_and_c_equivalence()
    test_pixelwise_median()
