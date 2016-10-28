# The Robovision Multiple View Stereo (MVS) Benchmark

# WARNING!!! This benchmark is still under active development! 

# Introduction

This directory contains code for benchmarking the performance of 3D reconstruction algorithms. Goals:

1. Make it easier to compare open source and commercial MVS reconstruction algorithms
1. Enable tuning of the speed vs. completeness vs. accuracy tradeoff
1. Focus on setups most relevant to Robovision's industrial applications, i.e. run on Robovision's data

More details:

1. The focus on synchronized inward-facing camera arrays imaging a single object a single time.
1. Camera calibration is assumed to be a solved problem; intrinsics and extrinsics will be provided. If you suspect that the provided camera parameters are sub-optimal, please let us know about it. Of course you may re-compute your own camera parameters if that is more convenient in your pipeline than loading ours. As long as they are saved and re-loaded, that doesn't count against the benchmark time.
1. These days some pipelines are computing a sparse point cloud using feature matching, and using that both to get the camera positions and also to bootstrap the 3D reconstruction. If your pipeline does this, the generation of the features and sparse point cloud must be included in the reported reconstruction time. In general, any computations that depend on the images being used for reconstruction must be counted in the reconstruction time. 
1. Any parameters you tune by hand must be held constant between reconstructions. Any parameters you tune automatically must be held constant, otherwise their compuation must be included in the reconstruction time.

For context, this benchmark is similar to the Middlebury Multi-View Stereo Benchmark: http://vision.middlebury.edu/mview/. Some differences:
1. We have higher resolution images
1. Middlebury includes benchmarks with many views; our setup is most similar to "dino sparse ring" case
1. Middlebury used a laser scanner and a spherical gantry to generate high quality ground truth data. Our ground truth will just be pmvs2 run with very high quality settings.
1. On the upside, our imgages are all captured simultaneously, so there is no illumination variation between shots, as there is with a spherical gantry
1. We do not have a fancy online interface for viewing results or submissions, etc... 

A note about rigor: this is NOT a research oriented benchmark. I am developing it in one day. Think of it as a tool for us to use in communication with vendors, that is at least better than looking at one point cloud and going "meh, looks good..."

## Dependencies
I used the default pcl in ubuntu 14.04:
sudo apt-get install libpcl-1.7-all-dev

C++ development stuff
sudo apt-get install build-essential
sudo apt-get install cmake cmake-curses-gui
sudo apt-get install git g++

python stuff
sudo apt-get install python3 python3-numpy

Warning! Dependencies are still in flux!

## Building

The benchmark is a mix of Python 3 and C++ wrapped in python. You must first build the C++ parts:

1. cd multi_view_stereo_benchmark
1. ccmake .
1. (press c to configure, e if PCL stuff throws errors, g to generate Makefiles and exit)
1. make

## Running

1. Download the dataset from the google drive and place it as a subdirectory "data" in the root directory of this git repository.
1. ./compare_clouds.py should run without errors in under a second on a typical workstation.

# FAQ

1. Why are the images upside down? Because my cameras are mounted upside down.
