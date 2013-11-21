#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/ros/conversions.h>
#include <pcl/features/vfh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler;


/** \brief create and save to disk a training data base of poses and corresponding vfh features
    \param -i the input pointcloud to transform
    \param -o the output directory
    \param -ntheta the number of steps about z-axis to calculate features for
    \param -nphi the number of steps about x-axis to calculate features for
*/
int main (int argc, char **argv)
{
    //parse input pointcloud
    std::string inputCloudName;
    pcl::console::parse_argument (argc, argv, "-i", inputCloudName );
    boost::filesystem::path inputCloudPath(inputCloudName);

    //parse output  pointcloud
    std::string outputDirName;
    pcl::console::parse_argument (argc, argv, "-o", outputDirName);
    boost::filesystem::path outputCloudDir(outputDirName);

    //parse number of theta views
    int i_numThetas = 10;
    pcl::console::parse_argument (argc, argv, "-ntheta", i_numThetas);
    size_t numThetas = int(i_numThetas);

    //parse number of theta views
    int i_numPhis = 10;
    pcl::console::parse_argument (argc, argv, "-nphi", i_numPhis);
    size_t numPhis = int(i_numPhis);

    pcl::console::print_highlight ("Creating %d views of the original object .\n", numThetas*numPhis); 

    PointCloud::Ptr cloud (new PointCloud);
    PointCloud::Ptr cloudRot (new PointCloud);


    //read ply file
    pcl::PolygonMesh triangles;
    if(inputCloudPath.extension().native().compare(".ply") == 0)
    {
        if( pcl::io::loadPolygonFilePLY(inputCloudName, triangles) == -1)
        {
            PCL_ERROR("Could not read .ply file\n");
            return(-1);
        }
        pcl::fromPCLPointCloud2(triangles.cloud, *cloud);
    }
    //read pcd file
    else if(inputCloudPath.extension().native().compare(".pcd") == 0)
    {
        pcl::PCDReader reader;
        if( reader.read(inputCloudName, *cloud) == -1)
        {
            PCL_ERROR("Could not read .pcd file\n");
            return(-1);
        }
    }
    else
    {
        
        PCL_ERROR("File must have extension .ply or .pcd\n");
        return(-1);
    }

    //Move point cloud so it is is centered at the origin
    Eigen::Matrix<float,4,1> centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    pcl::demeanPointCloud(*cloud, centroid, *cloud);

    //loop over viewpoints    
    float theta;
    float phi;
    Eigen::Matrix3f rotation; 
    pcl::PCDWriter writer;
    for (size_t i = 0; i < numThetas; ++i)
    {
        theta = 2*M_PI * i / numThetas;
        for (size_t j = 0; j < numPhis; ++j)
        {
            phi = M_PI * j / numPhis;

            //transform pointcloud
            rotation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitX());
            pcl::transformPointCloud(*cloud, *cloudRot, Eigen::Affine3f(rotation));
            
            std::cout << "Theta = " << theta*180/M_PI << " degrees, Phi = " << phi*180/M_PI << " degrees." << std::endl;

           
            //save pointcloud
            std::string outputFileName = inputCloudPath.stem().native();
            outputFileName.append("_").append(boost::lexical_cast<std::string>(i))
                .append("_").append(boost::lexical_cast<std::string>(j)).append(".pcd");

            boost::filesystem::path outputCloudPath;
            outputCloudPath = outputCloudDir / boost::filesystem::path(outputFileName);

            std::cout << "saving: " << outputCloudPath << std::endl;
            writer.writeBinary(outputCloudPath.native(), *cloudRot);

            //save angle data in separate file
            std::string angleDataFileName = inputCloudPath.stem().native();
            angleDataFileName.append("_").append(boost::lexical_cast<std::string>(i))
                .append("_").append(boost::lexical_cast<std::string>(j)).append(".txt");

            boost::filesystem::path angleDataPath;
            angleDataPath = outputCloudDir / boost::filesystem::path(angleDataFileName);

            std::ofstream angleData;
            angleData.open (angleDataPath.c_str());
                angleData << theta << "\n";
                angleData << phi << "\n";
            angleData.close (); 

            //visualize point cloud
            /*
            pcl::visualization::PCLVisualizer visu("viewer");
            visu.addPointCloud<pcl::PointXYZ> (cloudRot, ColorHandler(cloud, 0.0 , 255.0, 0.0), "cloud");
            visu.addCoordinateSystem (0.01, 0);
            visu.spinOnce(500);
            */
        }
    }

    return 0;
}
