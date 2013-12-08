#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
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
#include <pcl/console/parse.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <list>
#include "vfhLib.h"

struct VFHModel
{
    float theta; //angle about z axis
    float phi; // angle about x axis
    boost::filesystem::path filePath;
    float hist[histLength];
};

/** \brief loads either angle data corresponding  
    \param path path to .txt file containing angle information
    \param vfhModel stuct to load theta and phi angles into
*/
bool loadCloudAngleData(const boost::filesystem::path &path, VFHModel &vfhModel)
{
    //open file
    std::cout << "Loading: " << path.filename() << std::endl;
    ifstream fs;
    fs.open (path.c_str());
    if (!fs.is_open () || fs.fail ())
        return false;

    //load angle data
    std::string angle;
    std::getline (fs, angle, ' ');
    vfhModel.theta = boost::lexical_cast<float>(angle);
    std::getline (fs, angle);
    vfhModel.phi = boost::lexical_cast<float>(angle);
    fs.close ();

    //save filename
    vfhModel.filePath = path;
    vfhModel.filePath.replace_extension(".pcd");
    return true;
}   
    

/** \brief load object view points from input directory, generate
    \param -d the directory containing point clouds and corresponding angle data files 
*/
int main (int argc, char **argv)
{
    //parse data directory
    std::string dataDirName;
    pcl::console::parse_argument (argc, argv, "-d", dataDirName);
    boost::filesystem::path dataDir(dataDirName);
    

    //loop over all pcd files in the data directry and calculate vfh features
    PointCloud::Ptr cloud (new PointCloud);
    Eigen::Matrix<float,4,1> centroid;
    std::list<VFHModel> training; //training data list
    boost::filesystem::directory_iterator dirItr(dataDir), dirEnd;
    boost::filesystem::path angleDataPath;

    for(dirItr; dirItr != dirEnd; ++dirItr)
    {
        //skip txt and other files
        if(dirItr->path().extension().native().compare(".pcd") != 0)
            continue;

        //load point cloud
        if(!loadPointCloud(dirItr->path(), *cloud))
            return -1;
        
        //load angle data from txt file
        angleDataPath = dirItr->path();
        angleDataPath.replace_extension(".txt");
        VFHModel vfhModel;
        if(!loadCloudAngleData(angleDataPath, vfhModel))
            return -1;

        //Move point cloud so it is is centered at the origin
        /**
        pcl::compute3DCentroid(*cloud, centroid);
        pcl::demeanPointCloud(*cloud, centroid, *cloud);
        */

        //setup normal estimation class
        Normals::Ptr normals (new Normals);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new pcl::search::KdTree<pcl::PointXYZ>);
        normEst.setSearchMethod(normTree);
        normEst.setRadiusSearch(0.003);

        //estimate normals
        normEst.setInputCloud(cloud);
        normEst.compute(*normals);

        //Create VFH estimation class
        pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new pcl::search::KdTree<pcl::PointXYZ>);
        vfh.setSearchMethod(vfhsTree);
        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>);
        vfh.setViewPoint(0, 0, 0);

        //compute vfhs features
        vfh.setInputCloud(cloud);
        vfh.setInputNormals(normals);
        vfh.compute(*vfhs);

        //store vfhs feature in vfh model and push it to the training data list
        for(size_t i = 0; i < histLength; ++i)
        {
            vfhModel.hist[i] = vfhs->points[0].histogram[i];
        }
        training.push_front(vfhModel);

        //visualize point cloud
        /*
        pcl::visualization::PCLVisualizer visu("viewer");
        visu.addPointCloud<pcl::PointXYZ> (cloud, ColorHandler(cloud, 0.0 , 255.0, 0.0), "cloud");
        visu.addCoordinateSystem (0.01, 0);

        //visualize histogram
        pcl::visualization::PCLHistogramVisualizer histvis;
        histvis.addFeatureHistogram<pcl::VFHSignature308> (*vfhs, histLength);
        visu.spinOnce (500);
        */
    }

    //convert training data to FLANN format
    flann::Matrix<float> data (new float[training.size() * histLength], training.size(), histLength);
    size_t i = 0;
    std::list<VFHModel>::iterator it;
    for(it = training.begin(); it != training.end(); ++it)
    {
        for(size_t j = 0; j < data.cols; ++j)
        {
            data[i][j] = it->hist[j];
        }
        ++i;
    }

    //filenames
    std::string featuresFileName = "training_features.h5";
    std::string anglesFileName = "training_angles.list";
    std::string kdtreeIdxFileName = "training_kdtree.idx";

    // Save features to data file 
    flann::save_to_file (data, featuresFileName, "training_data");

    // Save angles to data file
    std::ofstream fs;
    fs.open (anglesFileName.c_str ());
    for(it = training.begin(); it != training.end(); ++it)
    {
        fs << it->theta << " " << it->phi << " " << it->filePath.native() << "\n";
    }
    fs.close ();

    // Build the tree index and save it to disk
    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...", kdtreeIdxFileName.c_str (), (int)data.rows);
    flann::Index<flann::ChiSquareDistance<float> > index (data, flann::LinearIndexParams ());
    //flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
    index.buildIndex ();
    index.save (kdtreeIdxFileName);
    delete[] data.ptr ();
    pcl::console::print_error (stderr, "Done\n");

    return 0;
}
