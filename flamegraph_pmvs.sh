#!/usr/bin/env bash

# Generate a flamegraph to monitor pmvs2 performance.
# Re-runs using the configuration in working_directory_pmvs.
# This may or may not be the reconstruction you care about.

PMVS=$PWD/extern/CMVS-PMVS/program/main/pmvs2
FLAMEGRAPH=$PWD/extern/FlameGraph

#(
#cd working_directory_pmvs
#sudo perf record -F 1000 -ag -- $PMVS ./ option.txt
#sudo chmod 777 perf.data
##perf script | $FLAMEGRAPH/stackcollapse-perf.pl > out.perf-folded
#perf script | c++filt | $FLAMEGRAPH/stackcollapse-perf.pl > out.perf-folded
#$FLAMEGRAPH/flamegraph.pl out.perf-folded > pmvs2_flamegraph.svg
#google-chrome pmvs2_flamegraph.svg
##c++filt < out.perf-folded > out.perf-demangled
#)

(
cd working_directory_pmvs
valgrind --tool=callgrind $PMVS ./ option.txt
)
