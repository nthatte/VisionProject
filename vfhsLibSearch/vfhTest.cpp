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

typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
//typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler;

const int histLength = 308;

/** \brief Load the list of angles from an ASCII file
  * \param list of angle
  * \param filename the input file name
  */
bool loadFileList (std::vector<float> &angles, const std::string &filename)
{
    ifstream fs;
    fs.open (filename.c_str ());
    if (!fs.is_open () || fs.fail ())
        return (false);

    std::string line;
    while (!fs.eof ())
    {
        std::getline (fs, line);
        if (line.empty ())
            continue;
        std::string angleStr = line;
        float angle = boost::lexical_cast<float>(angleStr);
        angles.push_back (angle);
    }
    fs.close ();
    return (true);
}

/** \brief Search for the closest k neighbors
  * \param index the tree
  * \param vfhs pointer to the query vfh feature
  * \param k the number of neighbors to search for
  * \param indices the resultant neighbor indices
  * \param distances the resultant neighbor distances
  */
inline void
nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
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

/** \brief Returns closest poses of of objects in training data to the query object
    \param -k the number of nearest neighbors to return
    \param -theta the query object pose angle
*/
int main (int argc, char **argv)
{
    int k = 1;
    pcl::console::parse_argument (argc, argv, "-k", k);
    pcl::console::print_highlight ("using %d nearest neighbors.\n", k); 

    float theta = 0.0f;
    pcl::console::parse_argument (argc, argv, "-theta", theta);
    pcl::console::print_highlight ("Query Angle = %f degrees.\n", theta); 
    theta = theta*M_PI/180;

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

    //Estimate normals
    Normals::Ptr normals (new Normals);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
    normEst.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new pcl::search::KdTree<pcl::PointXYZ>);
    normEst.setSearchMethod(normTree);
    normEst.setRadiusSearch(0.005);
    normEst.compute(*normals);

    //Visualize point cloud
    /*
    pcl::visualization::PCLVisualizer visu("viewer");
    visu.addPointCloud<pcl::PointXYZ> (cloud, ColorHandler(cloud, 0.0 , 255.0, 0.0), "cloud");
    visu.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 5, 0.01f, "normals", 0);
    visu.addCoordinateSystem (0.01, 0);
    */

    //Create VFH estimation class
    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud(cloud);
    vfh.setInputNormals(normals);
    vfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr vfhsTree (new pcl::search::KdTree<pcl::PointXYZ>);
    vfh.setSearchMethod(vfhsTree);

    //Rotate object to query pose and calculate VFHS features
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>);
    Eigen::Vector3f viewPointOrig(1.0, 0.0, 0.0);
    Eigen::Vector3f viewPoint;
    Eigen::Matrix3f rotation; 
    rotation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ());
    viewPoint = rotation*viewPointOrig;
    vfh.setViewPoint(viewPoint(0), viewPoint(1), viewPoint(2));
    vfh.compute(*vfhs);
    
    //filenames
    std::string kdtree_idx_file_name = "kdtree.idx";
    std::string training_data_h5_file_name = "training_data.h5";
    std::string training_data_list_file_name = "training_data.list";

    //allocate flann matrices
    std::vector<float> angles;
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;
    flann::Matrix<float> data;

    //load training data angles list
    loadFileList(angles, training_data_list_file_name);
    flann::load_from_file (data, training_data_h5_file_name, "training_data");
    flann::Index<flann::ChiSquareDistance<float> > index (data, flann::SavedIndexParams ("kdtree.idx"));

    //perform knn search
    index.buildIndex ();
    nearestKSearch (index, vfhs, k, k_indices, k_distances);

    // Output the results on screen
    pcl::console::print_highlight ("The closest %d neighbors are:\n", k);
    for (int i = 0; i < k; ++i)
    {
        pcl::console::print_info ("    %d - %f (%d) with a distance of: %f\n", 
            i, angles.at(k_indices[0][i])*180.0/M_PI, k_indices[0][i], k_distances[0][i]);
    }

    /*visualize histogram
    pcl::visualization::PCLHistogramVisualizer histvis;
    histvis.spin();
    */
    return 0;
}
