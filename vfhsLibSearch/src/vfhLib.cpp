#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/ros/conversions.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include "vfhLib.h"

/** \brief loads either a .pcd or .ply file into a pointcloud 
    \param cloud pointcloud to load data into
    \param path path to pointcloud file
*/
bool loadPointCloud(const boost::filesystem::path &path, PointCloud &cloud)
{
    std::cout << "Loading: " << path.filename() << std::endl;
    //read ply file
    pcl::PolygonMesh triangles;
    if(path.extension().native().compare(".ply") == 0)
    {
        if( pcl::io::loadPolygonFilePLY(path.native(), triangles) == -1)
        {
            PCL_ERROR("Could not read .ply file\n");
            return false;
        }
#if PCL17
        pcl::fromPCLPointCloud2(triangles.cloud, cloud);
#endif
#if PCL16
        pcl::fromROSMsg(triangles.cloud, cloud);
#endif
    }
    //read pcd file
    else if(path.extension().native().compare(".pcd") == 0)
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
        
        PCL_ERROR("File must have extension .ply or .pcd\n");
        return false;
    }
    return true;
}

struct CloudInfo
{
    float roll; 
    float pitch;
    float yaw; 
    boost::filesystem::path filePath;
};

/** \brief Load the list of angles from an ASCII file
  * \param list of angle
  * \param filename the input file name
  */
static bool loadAngleData (std::vector<CloudInfo> &cloudInfoList, const std::string &filename)
{
    ifstream fs;
    fs.open (filename.c_str ());
    if (!fs.is_open () || fs.fail ())
        return false;

    CloudInfo cloudinfo;
    std::string line;
    while (!fs.eof ())
    {
        //read roll
        std::getline (fs, line, ' ');
        if (line.empty ())
            continue;
        cloudinfo.roll = boost::lexical_cast<float>(line);

        //read pitch
        std::getline (fs, line, ' ');
        if (line.empty ())
            continue;
        cloudinfo.pitch = boost::lexical_cast<float>(line);

        //assign yaw
        cloudinfo.yaw = 0;

        //read filename
        std::getline (fs, line);
        if (line.empty ())
            continue;
        cloudinfo.filePath = boost::filesystem::path(line);
        cloudinfo.filePath.replace_extension(".pcd");
        cloudInfoList.push_back (cloudinfo);
    }
    fs.close ();
    return true;
}

/** \brief Search for the closest k neighbors
  * \param index the tree
  * \param vfhs pointer to the query vfh feature
  * \param k the number of neighbors to search for
  * \param indices the resultant neighbor indices
  * \param distances the resultant neighbor distances
  */
static void nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
    //store in flann query point
    flann::Matrix<float> p = flann::Matrix<float>(new float[histLength], 1, histLength);
    for(size_t i = 0; i < histLength; ++i)
    {
        p[0][i] = vfhs->points[0].histogram[i];
    }

    indices = flann::Matrix<int>(new int[k], 1, k);
    distances = flann::Matrix<float>(new float[k], 1, k);
    index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
    delete[] p.ptr ();
}

/** \brief Returns closest pose of closest cloud in training dataset to the query cloud
    \param cloud the query point cloud
    \param roll roll angle
    \param pitch pitch angle
    \param yaw yaw angle
    \param visMatch whether or not to visualze the closest match
*/
bool getPose (const PointCloud::Ptr &cloud, float &roll, float &pitch, float &yaw, const bool visMatch)
{
    //Estimate normals
    Normals::Ptr normals (new Normals);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
    normEst.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new pcl::search::KdTree<pcl::PointXYZ>);
    normEst.setSearchMethod(normTree);
    normEst.setRadiusSearch(0.005);
    normEst.compute(*normals);

    //Create VFH estimation class
    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new pcl::search::KdTree<pcl::PointXYZ>);
    vfh.setSearchMethod(vfhsTree);

    //calculate VFHS features
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>);
    vfh.setViewPoint(0, 0, 0);
    vfh.compute(*vfhs);
    
    //filenames
    std::string featuresFileName = "training_features.h5";
    std::string anglesFileName = "training_angles.list";
    std::string kdtreeIdxFileName = "training_kdtree.idx";

    //allocate flann matrices
    std::vector<CloudInfo> cloudInfoList;
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;
    flann::Matrix<float> data;

    //load training data angles list
    if(!loadAngleData(cloudInfoList, anglesFileName))
        return false;
    flann::load_from_file (data, featuresFileName, "training_data");
    flann::Index<flann::ChiSquareDistance<float> > index (data, flann::SavedIndexParams ("training_kdtree.idx"));

    //perform knn search
    index.buildIndex ();
    nearestKSearch (index, vfhs, 1, k_indices, k_distances);
    roll  = cloudInfoList.at(k_indices[0][0]).roll;
    pitch = cloudInfoList.at(k_indices[0][0]).pitch;
    yaw   = cloudInfoList.at(k_indices[0][0]).yaw; 

    // Output the results on screen
    if(visMatch)
    {
        pcl::console::print_highlight ("The closest neighbor is:\n");
        pcl::console::print_info ("roll = %f, pitch = %f, yaw = %f,  (%s) with a distance of: %f\n", 
            roll*180.0/M_PI, pitch*180.0/M_PI, yaw*180.0/M_PI, 
            cloudInfoList.at(k_indices[0][0]).filePath.c_str(), 
            k_distances[0][0]);

        //retrieve matched pointcloud
        PointCloud::Ptr cloudMatch (new PointCloud);
        pcl::PCDReader reader;
        reader.read(cloudInfoList.at(k_indices[0][0]).filePath.native(), *cloudMatch);

        //Move point cloud so it is is centered at the origin
        Eigen::Matrix<float,4,1> centroid;
        pcl::compute3DCentroid(*cloudMatch, centroid);
        pcl::demeanPointCloud(*cloudMatch, centroid, *cloudMatch);

        //visualize histogram
        /*
        pcl::visualization::PCLHistogramVisualizer histvis;
        histvis.addFeatureHistogram<pcl::VFHSignature308> (*vfhs, histLength);
        */

        //Visualize point cloud and matches
        //viewpoint calcs
        int y_s = (int)std::floor (sqrt (2.0));
        int x_s = y_s + (int)std::ceil ((2.0 / (double)y_s) - y_s);
        double x_step = (double)(1 / (double)x_s);
        double y_step = (double)(1 / (double)y_s);
        int viewport = 0, l = 0, m = 0;

        //setup visualizer and add query cloud 
        pcl::visualization::PCLVisualizer visu("KNN search");
        visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);

        //Move point cloud so it is is centered at the origin
        PointCloud::Ptr cloudDemeaned (new PointCloud);
        pcl::compute3DCentroid(*cloud, centroid);
        pcl::demeanPointCloud(*cloud, centroid, *cloudDemeaned);
        visu.addPointCloud<pcl::PointXYZ> (cloudDemeaned, ColorHandler(cloud, 0.0 , 255.0, 0.0), "Query Cloud Cloud", viewport);

        visu.addText ("Query Cloud", 20, 30, 136.0/255.0, 58.0/255.0, 1, "Query Cloud", viewport); 
        visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, "Query Cloud", viewport);
        visu.addCoordinateSystem (0.05, 0);

        //add matches to plot
        //shift viewpoint
        ++l;

        //names and text labels
        std::string viewName = "match";
        std::string textString = viewName;
        std::string cloudname = viewName;

        //add cloud
        visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
        visu.addPointCloud<pcl::PointXYZ> (cloudMatch, ColorHandler(cloudMatch, 0.0 , 255.0, 0.0), cloudname, viewport);
        visu.addText (textString, 20, 30, 136.0/255.0, 58.0/255.0, 1, textString, viewport);
        visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, textString, viewport);
        visu.spin();
    }

    return true;
}
