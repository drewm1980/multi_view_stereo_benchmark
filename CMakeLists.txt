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

#find_package(PCL 1.7 REQUIRED COMPONENTS common io visualization sample_consensus)
find_package(PCL 1.8 REQUIRED COMPONENTS common io visualization sample_consensus)
#target_link_directories(${PCL_LIBRARY_DIRS})
#target_add_definitions(${PCL_DEFINITIONS})

add_library(compare_clouds SHARED compare_clouds.cpp)
target_include_directories(compare_clouds PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(compare_clouds ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES})
target_compile_options(compare_clouds PRIVATE -std=c++11 -O3 -Werror -Wall -ffast-math)

#add_executable(compare_clouds_bin compare_clouds.cpp)
#set_target_properties(compare_clouds_bin PROPERTIES OUTPUT_NAME compare_clouds)
#target_include_directories(compare_clouds_bin PRIVATE ${PCL_INCLUDE_DIRS})
#target_link_libraries(compare_clouds_bin ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES})
#target_compile_options(compare_clouds_bin PRIVATE -std=c++11 -O3 -Werror -Wall -ffast-math)
