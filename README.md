# The Robovision Multiple View Stereo (MVS) Benchmark

## Introduction

This directory contains code for benchmarking the performance of 3D reconstruction algorithms. Goals:

1. Make it easier to compare open source and commercial MVS reconstruction algorithms
1. Enable tuning of the speed vs. completeness vs. accuracy tradeoff
1. Focus on setups most relevant to Robovision's industrial applications, i.e. run on Robovision's data
1. Improve reproduceability by actually integrating public reconstruction codes in the benchmark

More details:

1. The focus on synchronized inward-facing camera arrays imaging a single object a single time.
1. Camera calibration is assumed to be a solved problem; intrinsics and extrinsics will be provided. If you suspect that the provided camera parameters are sub-optimal, please let us know about it. Of course you may re-compute your own camera parameters if that is more convenient in your pipeline than loading ours. As long as they are saved and re-loaded, that doesn't count against the benchmark time.
1. These days some pipelines are computing a sparse point cloud using feature matching, and using that both to get the camera positions and also to bootstrap the 3D reconstruction. If your pipeline does this, the generation of the features and sparse point cloud must be included in the reported reconstruction time. In general, any computations that depend on the images being used for reconstruction must be counted in the reconstruction time. 
1. Any parameters you tune by hand must be held constant between reconstructions. Any parameters you tune automatically must be held constant, otherwise their compuation must be included in the reconstruction time.

For context, this benchmark is similar to the Middlebury Multi-View Stereo Benchmark: http://vision.middlebury.edu/mview/. Some differences:

1. We have higher resolution images
1. Middlebury includes benchmarks with many views; our setup is most similar to "dino sparse ring" case
1. Middlebury used a laser scanner and a spherical gantry to generate high quality ground truth data. Our ground truth will just be pmvs2 run with very high quality settings, with outliers removed manually.
1. On the upside, our imgages are all captured simultaneously, so there is no illumination variation between shots, as there is with a spherical gantry
1. We do not have a fancy online interface for viewing results or submissions, etc... 

A note about rigor: this is NOT a research oriented benchmark. It only aspires to be more rigorous than evaluating point clouds just by looking at them.

## Dependencies

These instructions are for Ubuntu 14.04

C++ development stuff
    `sudo apt-get install build-essential`
    `sudo apt-get install cmake cmake-curses-gui`
    `sudo apt-get install git g++`

python stuff
    `sudo apt-get install python3 python3-numpy python3-pandas python3-scipy python3-networkx`

I was able to run the code with the same dependencies installed with a conda python3 environment on Ubuntu 16.04 today (July 3 2017).

## Building

1. Clone everything repo using `git clone --recursive git@github.com:drewm1980/multi_view_stereo_benchmark.git`.  This will pull down the code for the benchmark itself, the dataset, and all of the dependencies.  

1. The benchmark is a mix of Python 3 and C++ wrapped in python. You must first build the C++ parts:
    `cd multi_view_stereo_benchmark`
    `ccmake .`
    (press `c` to configure, `e` if PCL stuff throws errors, `g` to generate Makefiles and exit)
    `make`

1. Build the reconstruction algorithms by running bootstrap.sh. See extern/README.md for details.

## Running
1. Run ipython3 (or just ipython if you have a conda 3 environment enabled) in the multi_view_stereo_benchmark directory
1. type `cd ..`
1. type `import multi_view_stereo_benchmark`
1. type `cd multi_view_stereo_benchmark`
1. type `run reconstruct.py` to generate all of the point cloud reconstructions
1. type `run benchmark.py` to perform all of point cloud comparisons and output the benchmark results

Note: If you're having import path problems, you're not alone. Read http://python-notes.curiousefficiency.org/en/latest/python_concepts/import_traps.html and double-check that you followed the above steps exactly. 

## FAQ

1. Why are the images upside down? Because my cameras are mounted upside down.
1. Why is the directory structure the way it is? This benchmark evolved out of another internal benchmark and I'm trying to keep it mostly compatible.
1. Why are the images so dark/underexposed? Exposure is locked down on my rig, and we also image some more reflective objects that are not in the benchmark. In general, saturation in overexposed images is worse than low SNR in underexposed images, so I err on the side of dark. Also these are machine vision cameras, not DSLR/smartphones; the data has not gone through any fancy image filters to make them look good.

## Current Status

Algorithms in the benchmark:

1. PMVS2, fully working
1. OpenCV StereoBM and StereoSGBM, fully working
1. gipuma, compiles and runs, full pipeline not working yet

Input formats in the benchmark:

1. PMVS2 format, complete
1. middlebury format, coded but not tested at all yet
1. HALCON camera calibration formats are used internally

WARNING! As far as I know, I'm the only user of this code. I regularly push accidentally broken versions to my public master branch during development, often in response to changing requirements of external non-public code. I can make tested snapshots on request.

## Licensing

The code for the benchmark itself is developed commercially by Robovision Integrated Solutons NV, and shared under the MIT license. See LICENSE.txt for the standard details. We shared it online to simplify some of our collaborations with partners. If you use it and like it, tell everyone how awesome we are. If you use it and hate it, tell nobody!

Each algorithm we include in the benchmark has its own license. At the time of writing, their licenses don't matter much because we run them as executables, not by linking (or calling) their API's.

