cmake_minimum_required(VERSION 2.8.12.2)
project(multi_view_stereo_benchmark)

set (CMAKE_CXX_FLAGS "-std=c++11")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
		"Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
		FORCE)
endif(NOT CMAKE_BUILD_TYPE)

find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
#include_directories( ${Boost_INCLUDE_DIR} )

find_package(PCL 1.7 REQUIRED COMPONENTS common io visualization sample_consensus)
#target_include_directories(compare_clouds PRIVATE ${PCL_INCLUDE_DIRS})
#target_link_directories(${PCL_LIBRARY_DIRS})
#target_add_definitions(${PCL_DEFINITIONS})

# Compile C functions used as inner loops in Python
add_library(compare_clouds SHARED compare_clouds.cpp)
target_link_libraries(compare_clouds ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES})
target_compile_options(compare_clouds PRIVATE -std=c++11 -O3 -Werror -Wall -ffast-math)
