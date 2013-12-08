#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/filesystem.hpp>

bool loadPointCloud(const boost::filesystem::path &path, pcl::PointCloud<pcl::PointXYZ> &cloud)
{
    std::cout << "Loading: " << path.filename() << std::endl;
    //read pcd file
    if(path.extension().native().compare(".pcd") == 0)
    {
        pcl::PCDReader reader;
        if( reader.read(path.native(), cloud) == -1)
        {
            PCL_ERROR("Could not read .pcd file\n");
            return false;
        }
    }
    else
    {
        
        PCL_ERROR("File must have extension .pcd\n");
        return false;
    }
    return true;
}

int main (int argc, char** argv)
{
    //read in point cloud
    std::string queryCloudName;
    pcl::console::parse_argument (argc, argv, "-c", queryCloudName);
    boost::filesystem::path queryCloudPath(queryCloudName);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn (new pcl::PointCloud<pcl::PointXYZ>);
    if(!loadPointCloud(queryCloudPath, *cloudIn))
        return -1;
 
    // Transform input cloud to obtain output
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut (new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Vector3f transvec(.1, 0, 0);
    Eigen::Translation<float, 3> trans = Eigen::Translation<float, 3>(transvec);
    Eigen::AngleAxisf rotZ = Eigen::AngleAxisf(35*(M_PI/180.0), Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf rotX = Eigen::AngleAxisf(10.0*(M_PI/180.0), Eigen::Vector3f::UnitX());
    pcl::transformPointCloud(*cloudIn, *cloudOut, Eigen::Affine3f(trans*rotX*rotZ);
    
    // setup 3dof icp 
    pcl::registration::WarpPointRigid3D<pcl::PointXYZ, pcl::PointXYZ>::Ptr wpr3d(new pcl::registration::WarpPointRigid3D<pcl::PointXYZ, pcl::PointXYZ>);
    pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>::Ptr te(new pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>);
    te->setWarpFunction(wpr3d);
    pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setTransformationEstimation(te);
    icp.setInputSource(cloudIn);
    icp.setInputTarget(cloudOut);

    // perform icp
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFinal(new pcl::PointCloud<pcl::PointXYZ>);
    icp.align(*cloudFinal);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
        icp.getFitnessScore() << std::endl;
        std::cout << icp.getFinalTransformation() << std::endl;

    //visualize results
    pcl::visualization::PCLVisualizer visu("icp results");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloudIn, 0, 255, 0);
    visu.addPointCloud<pcl::PointXYZ> (cloudIn, green, "input cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloudOut, 255, 0, 0);
    visu.addPointCloud<pcl::PointXYZ> (cloudOut, red, "output cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloudFinal, 0, 0, 255);
    visu.addPointCloud<pcl::PointXYZ> (cloudFinal, blue, "final cloud");

    visu.addCoordinateSystem(0.1, 0);
    visu.spin();

    return 0;
}
