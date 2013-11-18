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
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <fstream>
#include <iostream>

typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler;

const int histLength = 308;
struct vfh_model
{
    float angle;
    float hist[histLength];
};

/** \brief create and save to disk a training data base of poses and corresponding vfh features
    \param -n the number of poses to calculate features for
*/
int main (int argc, char **argv)
{
    int i_numThetas = 360;
    pcl::console::parse_argument (argc, argv, "-n", i_numThetas);
    size_t numThetas = int(i_numThetas);
    pcl::console::print_highlight ("Training on %d angles.\n", numThetas); 

    PointCloud::Ptr cloudRAW (new PointCloud);
    PointCloud::Ptr cloud (new PointCloud);

    //read ply file
    pcl::PolygonMesh triangles;
    if( pcl::io::loadPolygonFilePLY(argv[1], triangles) == -1)
    {
        PCL_ERROR("Could not read file\n");
        return(-1);
    }
    pcl::fromPCLPointCloud2(triangles.cloud, *cloudRAW);

    //read pcd file
    /*
    pcl::PCDReader reader;
    if( reader.read(argv[1], *cloud) == -1)
    {
        PCL_ERROR("Could not read file\n");
        return(-1);
    }
    */

    //Move point cloud so it is is centered at the origin
    Eigen::Matrix<float,4,1> centroid;
    pcl::compute3DCentroid(*cloudRAW, centroid);
    pcl::demeanPointCloud(*cloudRAW, centroid, *cloud);

    //estimate normals
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
    vfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new pcl::search::KdTree<pcl::PointXYZ>);
    vfh.setSearchMethod(vfhsTree);

    //loop over viewpoints and calculate VFHS features
    float theta;
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>);
    std::vector<vfh_model> training; //training data list
    training.resize(numThetas);
    Eigen::Vector3f viewPointOrig(1.0, 0.0, 0.0);
    Eigen::Vector3f viewPoint;
    Eigen::Matrix3f rotation; 
    for (size_t i = 0; i < numThetas; ++i)
    {
        //compute vfhs features
        theta = 2*M_PI * i / numThetas;
        training[i].angle = theta;
        rotation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ());
        viewPoint = rotation*viewPointOrig;
        vfh.setViewPoint(viewPoint(0), viewPoint(1), viewPoint(2));
        vfh.compute(*vfhs);

        //store in training data list
        for(size_t j = 0; j < histLength; ++j)
        {
            training[i].hist[j] = vfhs->points[0].histogram[j];
        }
        
        std::cout << "Angle = " << training[i].angle*180/M_PI << " degrees" << std::endl;
        /*visualize histogram
        pcl::visualization::PCLHistogramVisualizer histvis;
        histvis.addFeatureHistogram<pcl::VFHSignature308> (*vfhs, histLength);
        histvis.spinOnce (1000);
        */
    }

    //convert training data to FLANN format
    flann::Matrix<float> data (new float[training.size() * histLength], training.size(), histLength);

    for(size_t i = 0; i < data.rows; ++i)
    {
        for(size_t j = 0; j < data.cols; ++j)
        {
            data[i][j] = training[i].hist[j];
        }
    }

    //filenames
    std::string kdtree_idx_file_name = "kdtree.idx";
    std::string training_data_h5_file_name = "training_data.h5";
    std::string training_data_list_file_name = "training_data.list";

    // Save data to disk (list of models)
    flann::save_to_file (data, training_data_h5_file_name, "training_data");
    std::ofstream fs;
    fs.open (training_data_list_file_name.c_str ());
    for (size_t i = 0; i < training.size (); ++i)
    {
        fs << training[i].angle << "\n";
    }
    fs.close ();

    // Build the tree index and save it to disk
    pcl::console::print_error ("Building the kdtree index (%s) for %d elements...", kdtree_idx_file_name.c_str (), (int)data.rows);
    flann::Index<flann::ChiSquareDistance<float> > index (data, flann::LinearIndexParams ());
    //flann::Index<flann::ChiSquareDistance<float> > index (data, flann::KDTreeIndexParams (4));
    index.buildIndex ();
    index.save (kdtree_idx_file_name);
    delete[] data.ptr ();
    pcl::console::print_error (stderr, "Done\n");

    //visualize point cloud
    pcl::visualization::PCLVisualizer visu("viewer");
    visu.addPointCloud<pcl::PointXYZ> (cloud, ColorHandler(cloud, 0.0 , 255.0, 0.0), "cloud");
    visu.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 5, 0.01f, "normals", 0);
    visu.addCoordinateSystem (0.01, 0);

    visu.spin();

    return 0;
}
