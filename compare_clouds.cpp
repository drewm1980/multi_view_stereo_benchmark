#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <string>

extern "C" {
void compare_clouds(const float* cloud1, const float* cloud2, int points1,
                    int points2) {
    // Compare two point clouds, stored compactly in xyzxyz... format
}
}

using namespace std;
using Point = pcl::PointXYZ;

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
            "data/reconstructions/2016_10_24__17_43_02/reference.ply";
        cloud2FileName = "data/reconstructions/2016_10_24__17_43_02/medium.ply";
    } else {
        cloud1FileName = argv[1];
        cloud2FileName = argv[2];
    }
    cout << "Running on clouds: " << endl
         << cloud1FileName << endl
         << cloud2FileName << endl;

    auto cloud1 = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);
    auto cloud2 = pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>);

    pcl::io::loadPLYFile<Point>("pmvs_result.ply", *cloud1);
    pcl::io::loadPLYFile<Point>("pmvs_result.ply", *cloud2);

    // if (cloud1->width < 20) {
    // throw std::runtime_error("Loaded point cloud contains almost no points!
    // Th");
    //}
    // if (cloud1->width < 20) {
    // throw std::runtime_error("Loaded point cloud contains almost no
    // points!");
    //}
}
