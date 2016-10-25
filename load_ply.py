#!/usr/bin/env python3

# Simple .ply file loader to avoid adding an unnecessary dependency.

import numpy
from pathlib import Path

def load_ply(filename, enableCaching=True):
    """ Load an ascii based .ply file.
        Inputs:
            filename -- string or path
            enableCaching -- bool, defaults to True
        Outputs:
            data -- row-major AOS numpy array.
            columnnames -- list of string
            columntypes -- list of string 
    """
    # Parse the header
    filename = str(filename)
    file = open(filename,'r')
    assert file.readline().strip() == 'ply'
    assert file.readline().strip() == 'format ascii 1.0'
    nextline = file.readline().strip()
    header_lines = 3
    while(nextline.split(' ')[0] == 'comment'):
        header_lines += 1
        nextline = file.readline().strip()

    assert nextline.split(' ')[0] == 'element'
    assert nextline.split(' ')[1] == 'vertex'
    expected_vertices = int(nextline.split(' ')[2])

    columntypes = []
    columnnames = []

    while(1):
        line = file.readline().strip().split(' ')
        if line[0] == 'property':
            columntypes.append(line[1])
            columnnames.append(line[2])
            header_lines += 1
        if line[0] == 'end_header':
            header_lines += 1
            break

    file.close()

    plyPath = Path(filename)
    plyTimestamp = plyPath.stat().st_mtime
    plyCachedPath = plyPath.with_suffix('.npy')
    if enableCaching and plyCachedPath.is_file() and plyCachedPath.stat().st_mtime > plyTimestamp:
        #print("Pickled point cloud is newer than ascii .ply file; loading it!")
        data = numpy.load(file=str(plyCachedPath))
    else:
        data = numpy.loadtxt(fname=filename,skiprows=header_lines)
        if enableCaching:
            #print("Pickle non-existent or older than .ply file; regenerating it!")
            numpy.save(arr=data,file=str(plyCachedPath))

    assert data.shape[0]==expected_vertices
    return data, columnnames, columntypes

