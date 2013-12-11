#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <iostream>
#include "vfhLib.h"

int main (int argc, char **argv)
{
    boost::filesystem::path CloudPathTable("../goldFishTest2/table_cluster.pcd");
    boost::filesystem::path CloudPathMatch("../goldFishTest2/post_sac.pcd");

    //read in point clouds
    PointCloud::Ptr cloudTable (new PointCloud);
    if(!loadPointCloud(CloudPathTable, *cloudTable))
        return -1;

    PointCloud::Ptr cloudMatch (new PointCloud);
    if(!loadPointCloud(CloudPathMatch, *cloudMatch))
        return -1;

    //Move point cloud so it is is centered at the origin
    Eigen::Matrix<float,4,1> centroid;
    pcl::compute3DCentroid(*cloudMatch, centroid);
    pcl::demeanPointCloud(*cloudMatch, centroid, *cloudMatch);
    pcl::compute3DCentroid(*cloudTable, centroid);
    pcl::demeanPointCloud(*cloudTable, centroid, *cloudTable);

    //setup visualizer and add query cloud 
    pcl::visualization::PCLVisualizer visu("Cloud Matching");
    visu.addPointCloud<pcl::PointXYZ> (cloudTable, ColorHandler(cloudTable, 3 , 28, 72), "Cloud Table", 0);
    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "Cloud Table");
    visu.addPointCloud<pcl::PointXYZ> (cloudMatch, ColorHandler(cloudMatch, 230 , 33, 23), "Cloud Match", 0);
    visu.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "Cloud Match");
    visu.setBackgroundColor(1.0, 1.0, 1.0);
    visu.spin();

    return 0;
}
