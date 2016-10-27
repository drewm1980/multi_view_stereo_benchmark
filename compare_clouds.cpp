#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <string>
#include <vector>

using namespace std;
using Point = pcl::PointXYZ;

pcl::PointCloud<pcl::PointXYZ>::Ptr copy_c_array_to_point_cloud(const float* array,
                                                           int points) {
    auto cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < points * 3; i += 3) {
        cloud->push_back(pcl::PointXYZ(array[i], array[i + 1], array[i + 2]));
    }
    return cloud;
}

extern "C" {
void compare_clouds(const float* cloud1, const float* cloud2, int points1,
                    int points2) {
    cout << points1 << endl;
    cout << points2 << endl;

    // Compare two point clouds, stored compactly in xyzxyz... format
    auto cloud1_pcl = copy_c_array_to_point_cloud(cloud1, points1);
    auto cloud2_pcl = copy_c_array_to_point_cloud(cloud2, points2);
}
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

    auto cloud1 = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);
    auto cloud2 = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);

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
}
