#include <string>
#include <iostream>
#include <vector>

#ifdef USE_PCL
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/octree/octree.h>
#endif

using namespace std;

#pragma pack(push)
#pragma pack()
struct PointCloudComparisonResult {
    // This must manually be kept synchronized with PointCloudComparisonFields!
    float numCloud1Points;
    float numCloud2Points;
    float distanceThreshold;
    float numCloud1PointsNearCloud2;
    float numCloud2PointsNearCloud1;
};
#pragma pack(pop)

#ifdef USE_PCL
using Point = pcl::PointXYZ;

// TODO maybe with another initializer this can compile into a memcpy...
pcl::PointCloud<pcl::PointXYZ>::Ptr
copy_c_array_to_point_cloud(const float* array, int points) {
    auto cloud =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < points * 3; i += 3) {
        cloud->push_back(pcl::PointXYZ(array[i], array[i + 1], array[i + 2]));
    }
    return cloud;
}
#endif

extern "C" {
using namespace std;
bool is_point_close_to_cloud_bruteforce(const float* point_, const float* cloud,
                                        int points, float distanceThreshold) {
    float point[3];
    point[3] = 0.0f;
    for (int i = 0; i < 3; i++) point[i] = point_[i];
    point[0] = point_[0];
    point[1] = point_[1];
    for (int cloudIndex = 0; cloudIndex < 3 * points; cloudIndex += 3) {
        float cloudpoint[3];
        for (int i = 0; i < 3; i++) cloudpoint[i] = cloud[cloudIndex + i];
        float distancePoint[3];
        for (int i = 0; i < 3; i++) distancePoint[i] = cloudpoint[i] - point[i];
        float squaredDistance = 0.0f;
        for (int i = 0; i < 3; i++)
            squaredDistance += distancePoint[i] * distancePoint[i];
        if (squaredDistance < distanceThreshold * distanceThreshold)
            return true;
    }
    return false;
}

void compare_clouds_bruteforce(const float* cloud1_, const float* cloud2_,
                               int points1, int points2,
                               float distanceThreshold,
                               PointCloudComparisonResult* result) {
    int numCloud1PointsNearCloud2 = 0;
    int numCloud2PointsNearCloud1 = 0;

    //cout << "searching for cloud1 points in cloud2...." << endl;
#pragma omp parallel for reduction(+ : numCloud1PointsNearCloud2)
    for (int i = 0; i < points1; i++) {
        if (is_point_close_to_cloud_bruteforce(cloud1_ + 3 * i, cloud2_,
                                               points2, distanceThreshold)) {
            numCloud1PointsNearCloud2++;
        }
    }
    //cout << numCloud1PointsNearCloud2 << endl;
    //cout << "searching for cloud2 points in cloud1...." << endl;
#pragma omp parallel for reduction(+ : numCloud2PointsNearCloud1)
    for (int i = 0; i < points2; i++) {
        if (is_point_close_to_cloud_bruteforce(cloud2_ + 3 * i, cloud1_,
                                               points1, distanceThreshold)) {
            numCloud2PointsNearCloud1++;
        }
    }
    //cout << numCloud2PointsNearCloud1 << endl;
    result->numCloud1Points = points1;
    result->numCloud2Points = points2;
    result->distanceThreshold = distanceThreshold;
    result->numCloud1PointsNearCloud2 = numCloud1PointsNearCloud2;
    result->numCloud2PointsNearCloud1 = numCloud2PointsNearCloud1;
}

#ifdef USE_PCL
void compare_clouds_btree(const float* cloud1_, const float* cloud2_,
                          int points1, int points2, float octreeResolution,
                          float distanceThreshold,
                          PointCloudComparisonResult* result) {
    //cout << points1 << endl;
    //cout << points2 << endl;

    // Compare two point clouds, stored compactly in xyzxyz... format
    //cout << "Copying point cloud data into PCL data structures..." << endl;
    auto cloud1 = copy_c_array_to_point_cloud(cloud1_, points1);
    auto cloud2 = copy_c_array_to_point_cloud(cloud2_, points2);

    // float octreeResolution = .005f; // m. Note: Tunable for performance
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree1(
        octreeResolution);
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree2(
        octreeResolution);

    // Setting the bounding box doesn't seem to affect runtime substantially.
    // double r = .05;
    // auto set_bounding_box = [r](decltype(octree1)
    // octree){octree.defineBoundingBox(-r,-r,-r,r,r,r);};
    // set_bounding_box(octree1);
    // set_bounding_box(octree2);

    octree1.setInputCloud(cloud1);
    octree2.setInputCloud(cloud2);
    octree1.addPointsFromInputCloud();
    octree2.addPointsFromInputCloud();

    // float distanceThreshold = .002f; // m Note: Affects the meaning of the
    // benchmark!
    int numCloud1PointsNearCloud2 = 0;
    int numCloud2PointsNearCloud1 = 0;

    auto pointNearCloud =
        [distanceThreshold](pcl::PointXYZ searchPoint,
                            decltype(octree1) octree) -> bool {
            std::vector<int> indeces;
            std::vector<float> squaredDistances;
            int neighborsFound = octree.nearestKSearch(searchPoint, 1, indeces,
                                                       squaredDistances);
            if (neighborsFound == 0) return false;
            return squaredDistances[0] < distanceThreshold * distanceThreshold;
            // Note: Could also try octree.radiusSearch. may be faster.
            // Really want radiusSearch with early termination.
        };

    // Note: It's not really necessary to have both octrees allocated at the
    // same time
    //cout << "searching for cloud1 points in cloud2...." << endl;
    for (auto& searchPoint : cloud1->points) {
        if (pointNearCloud(searchPoint, octree2)) numCloud1PointsNearCloud2++;
    }
    //cout << numCloud1PointsNearCloud2 << endl;
    //cout << "searching for cloud2 points in cloud1...." << endl;
    for (auto& searchPoint : cloud2->points) {
        if (pointNearCloud(searchPoint, octree1)) numCloud2PointsNearCloud1++;
    }
    //cout << numCloud2PointsNearCloud1 << endl;

    result->numCloud1Points = points1;
    result->numCloud2Points = points2;
    result->distanceThreshold = distanceThreshold;
    result->numCloud2PointsNearCloud1 = numCloud2PointsNearCloud1;
    result->numCloud1PointsNearCloud2 = numCloud1PointsNearCloud2;
}
#endif
}

int main(int argc, char** argv) {
    string cloud1FileName;
    string cloud2FileName;
    // If compiled as an executable, load the first two arguments as pcl files
    // and
    // compare them
    if (argc == 1) {
        cout << "No parameters passed, comparing two hard-coded paths for ply "
                "files for debugging!" << endl;
        cloud1FileName =
            "./data/reconstructions/2016_10_24__17_43_02/reference.ply";
        cloud2FileName =
            "./data/reconstructions/2016_10_24__17_43_17/high_quality.ply";
    } else {
        cloud1FileName = argv[1];
        cloud2FileName = argv[2];
    }
    cout << "Running on clouds: " << endl
         << cloud1FileName << endl
         << cloud2FileName << endl;

#ifdef USE_PCL
    auto cloud1 = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);
    auto cloud2 = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);

    // PCL's point cloud parser pukes on files exported from meshlab
    pcl::io::loadPLYFile<Point>(cloud1FileName.c_str(), *cloud1);
    pcl::io::loadPLYFile<Point>(cloud2FileName.c_str(), *cloud2);

// if (cloud1->width < 20) {
// throw std::runtime_error("Loaded point cloud contains almost no points!
// Th");
//}
// if (cloud1->width < 20) {
// throw std::runtime_error("Loaded point cloud contains almost no
// points!");
//}
#endif
    cout << "Not really doing anything in this main file! Run the python "
            "wrapper!" << endl;
}
