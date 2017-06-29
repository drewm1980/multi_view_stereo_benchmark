#!/usr/bin/env bash
#. /home/awagner/.bashrc
#python reconstruct_opencv.py
sudo cpufreq-set --governor performance
sudo PATH=$PATH PYTHONPATH=$PYTHONPATH chrt --rr 99 python3 reconstruct_opencv.py
sudo cpufreq-set --governor powersave
sudo cat working_directory_opencv/reconstruction_runtime.txt
