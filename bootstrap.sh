#!/usr/bin/env bash

# A hacky way to build all of the dependencies.

# TODO: implement something custom command in the parent cmake instance to build this stuff without resorting to non-portable shell scripts.

# TODO: scrape these out of cmake somehow
USE_PMVS2=1
USE_GIPUMA=0

# Generates the pmvs2 binary in extern/CMVS-PMVS/program/pmvs2
if [ $USE_PMVS2 ]
then
	(
	cd extern/CMVS-PMVS/program
	cmake .
	make
	)
fi

if [ $USE_GIPUMA ]
then
	(
	cd extern/fusibile
	cmake .
	make
	)
	(
	cd extern/gipuma
	cmake .
	make
	)
fi
