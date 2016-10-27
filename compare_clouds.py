#!/usr/bin/env python3

from pathlib import Path

"""Code for comparing point clouds"""

cloud1Path = Path("./data/reconstructions/2016_10_24__17_43_17/reference.ply")
cloud2Path = Path("./data/reconstructions/2016_10_24__17_43_17/high_quality.ply")

from load_ply import load_ply

cloud1PointData = load_ply(cloud1Path)[0][:,:3].copy()
cloud2PointData = load_ply(cloud2Path)[0][:,:3].copy()

#if __name__=='__main__':
    #pass
