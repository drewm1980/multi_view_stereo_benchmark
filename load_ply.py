#!/usr/bin/env python3

# Simple .ply file loader to avoid adding an unnecessary dependency.

import pathlib
from pathlib import Path

import numpy

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

    header_lines = 0

    assert file.readline().strip() == 'ply'
    header_lines += 1

    assert file.readline().strip() == 'format ascii 1.0'
    header_lines += 1

    nextline = file.readline().strip()
    while(nextline.split(' ')[0] == 'comment'):
        header_lines += 1
        nextline = file.readline().strip()

    assert nextline.split(' ')[0] == 'element'
    assert nextline.split(' ')[1] == 'vertex'
    expected_vertices = int(nextline.split(' ')[2])
    header_lines += 1

    columntypes = []
    columnnames = []

    while(1):
        nextline = file.readline().strip().split(' ')
        if nextline[0] == 'property':
            columntypes.append(nextline[1])
            columnnames.append(nextline[2])
            header_lines += 1
            continue
        else:
            break

    # meshlab annoyingly exports files with zero faces, instead of just ommitting the element.
    if nextline[0] == 'element' and nextline[1] == 'face':
        assert nextline[2] == '0', "Nonzero number of faces in the ply file is not handled yet!"
        header_lines += 1

        nextline = file.readline().strip().split(' ')
        assert nextline[0] == 'property'
        header_lines += 1

        nextline = file.readline().strip().split(' ')
        
    assert nextline[0] == 'end_header'
    header_lines += 1

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
