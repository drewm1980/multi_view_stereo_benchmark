# The Robovision Multiple View Stereo (MVS) Benchmark

## Introduction

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

C++ development stuff
    `sudo apt-get install build-essential`
    `sudo apt-get install cmake cmake-curses-gui`
    `sudo apt-get install git g++`

python stuff
    `sudo apt-get install python3 python3-numpy python3-pandas`

Optional:
For one of my point cloud comparison implementations, I used the default pcl in ubuntu 14.04:
    `sudo apt-get install libpcl-1.7-all-dev`
At the time of writing you can ignore that.

## Building

1. Clone everything repo using `git clone --recursive git@github.com:drewm1980/multi_view_stereo_benchmark.git`.  This will pull down the code for the benchmark itself, the dataset, and all of the dependencies.  

1. The benchmark is a mix of Python 3 and C++ wrapped in python. You must first build the C++ parts:
    `cd multi_view_stereo_benchmark`
    `ccmake .`
    (press `c` to configure, `e` if PCL stuff throws errors, `g` to generate Makefiles and exit)
    `make`

1. Build the reconstruction algorithms by running bootstrap.sh. See extern/README.md for details.

## Running
1. Run `./reconstruct.py` to generate all of the point cloud reconstructions
1. Run `./benchmark.py` to perform all of point cloud comparisons and output the benchmark results

## FAQ

1. Why are the images upside down? Because my cameras are mounted upside down.
1. Why is the directory structure the way it is? This benchmark evolved out of another internal benchmark and I'm trying to keep it mostly compatible.
