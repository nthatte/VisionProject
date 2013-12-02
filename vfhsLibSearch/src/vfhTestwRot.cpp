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

typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler;

const int histLength = 308;

/** \brief Load the list of angles from an ASCII file
  * \param list of angle
  * \param filename the input file name
  */
struct CloudInfo
{
    float theta; //angle about z axis
    float phi; // angle about x axis
    boost::filesystem::path filePath;
};

bool loadAngleData (std::vector<CloudInfo> &cloudInfoList, const std::string &filename)
{
    ifstream fs;
    fs.open (filename.c_str ());
    if (!fs.is_open () || fs.fail ())
        return (false);

    CloudInfo cloudinfo;
    std::string line;
    while (!fs.eof ())
    {
        //read theta
        std::getline (fs, line, ' ');
        if (line.empty ())
            continue;
        cloudinfo.theta = boost::lexical_cast<float>(line);

        //read phi
        std::getline (fs, line, ' ');
        if (line.empty ())
            continue;
        cloudinfo.phi = boost::lexical_cast<float>(line);

        //read filename
        std::getline (fs, line);
        if (line.empty ())
            continue;
        cloudinfo.filePath = boost::filesystem::path(line);
        cloudinfo.filePath.replace_extension(".pcd");
        cloudInfoList.push_back (cloudinfo);
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
void nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, 
                     pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, 
                     int k, flann::Matrix<int> &indices, 
                     flann::Matrix<float> &distances) {
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
    \param -q the path to the input point cloud
    \param -k the number of nearest neighbors to return
    \param -theta the query angle about the z axis in degrees
    \param -phi the query angle about the x axis in degrees
*/
int main (int argc, char **argv)
{
    //parse data directory
    std::string queryCloudName;
    pcl::console::parse_argument (argc, argv, "-q", queryCloudName);
    boost::filesystem::path queryCloudPath(queryCloudName);

    //parse number of nearest neighbors k
    int k = 1;
    pcl::console::parse_argument (argc, argv, "-k", k);
    pcl::console::print_highlight ("using %d nearest neighbors.\n", k); 

    //read theta argument
    float theta;
    if(pcl::console::parse_argument (argc, argv, "-theta", theta) == -1)
    {
        PCL_ERROR("You must specify a query theta angle.\n");
        return -1;
    }

    //read phi argument
    float phi;
    if(pcl::console::parse_argument (argc, argv, "-phi", phi) == -1)
    {
        PCL_ERROR("You must specify a query phi angle");
        return -1;
    }
    pcl::console::print_highlight ("Query theta = %f degrees.\n", theta); 
    pcl::console::print_highlight ("Query phi = %f degrees.\n", phi); 
    theta *= M_PI/180;
    phi   *= M_PI/180;


    //read in point cloud
    PointCloud::Ptr cloud (new PointCloud);
    pcl::PCDReader reader;
    //read ply file
    pcl::PolygonMesh triangles;
    if(queryCloudPath.extension().native().compare(".ply") == 0)
    {
        if( pcl::io::loadPolygonFilePLY(queryCloudPath.native(), triangles) == -1)
        {
            PCL_ERROR("Could not read .ply file\n");
            return -1;
        }
#if PCL17
        pcl::fromPCLPointCloud2(triangles.cloud, *cloud);
#endif
#if PCL16
        pcl::fromROSMsg(triangles.cloud, *cloud);
#endif
    }
    //read pcd file
    else if(queryCloudPath.extension().native().compare(".pcd") == 0)
    {
        if( reader.read(queryCloudPath.native(), *cloud) == -1)
        {
            PCL_ERROR("Could not read .pcd file\n");
            return -1;
        }
    }
    else
    {
        
        PCL_ERROR("File must have extension .ply or .pcd\n");
        return -1;
    }

    //Move point cloud so it is is centered at the origin
    Eigen::Matrix<float,4,1> centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    pcl::demeanPointCloud(*cloud, centroid, *cloud);

    //Rotate object to query pose
    Eigen::Matrix3f rotation; 
    std::cout << theta << " " << phi << std::endl;
    rotation = Eigen::AngleAxisf(theta, Eigen::Vector3f::UnitZ()) * 
        Eigen::AngleAxisf(phi, Eigen::Vector3f::UnitX());
    pcl::transformPointCloud(*cloud, *cloud, Eigen::Affine3f(rotation));

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
    vfh.setViewPoint(1, 0, 0);
    vfh.compute(*vfhs);
    
    //filenames
    std::string featuresFileName = "training_features.h5";
    std::string anglesFileName = "training_angles.list";
    std::string kdtreeIdxFileName = "training_kdtree.idx";

    //allocate flann matrices
    std::vector<CloudInfo> cloudInfoList;
    std::vector<PointCloud::Ptr> cloudList;
    cloudList.resize(k);
    flann::Matrix<int> k_indices;
    flann::Matrix<float> k_distances;
    flann::Matrix<float> data;

    //load training data angles list
    loadAngleData(cloudInfoList, anglesFileName);
    flann::load_from_file (data, featuresFileName, "training_data");
    flann::Index<flann::ChiSquareDistance<float> > index (data, flann::SavedIndexParams ("training_kdtree.idx"));

    //perform knn search
    index.buildIndex ();
    nearestKSearch (index, vfhs, k, k_indices, k_distances);

    // Output the results on screen
    pcl::console::print_highlight ("The closest %d neighbors are:\n", k);
    for (int i = 0; i < k; ++i)
    {
        //print nearest neighbor info to screen
        pcl::console::print_info ("    %d - theta = %f, phi = %f  (%d) with a distance of: %f\n", 
            i, 
            cloudInfoList.at(k_indices[0][i]).theta*180.0/M_PI, 
            cloudInfoList.at(k_indices[0][i]).phi*180.0/M_PI, 
            k_indices[0][i],
            k_distances[0][i]);

        //retrieve pointcloud and store in list
        PointCloud::Ptr cloudMatch (new PointCloud);
        reader.read(cloudInfoList.at(k_indices[0][i]).filePath.native(), *cloudMatch);

        //Move point cloud so it is is centered at the origin
        pcl::compute3DCentroid(*cloudMatch, centroid);
        pcl::demeanPointCloud(*cloudMatch, centroid, *cloudMatch);

        cloudList[i] = cloudMatch;
    }

    //visualize histogram
    /*
    pcl::visualization::PCLHistogramVisualizer histvis;
    histvis.addFeatureHistogram<pcl::VFHSignature308> (*vfhs, histLength);
    */

    //Visualize point cloud and matches
    //viewpoint cals
    int y_s = (int)std::floor (sqrt ((double)k));
    int x_s = y_s + (int)std::ceil ((k / (double)y_s) - y_s);
    double x_step = (double)(1 / (double)x_s);
    double y_step = (double)(1 / (double)y_s);
    int viewport = 0, l = 0, m = 0;
    std::string viewName = "match number ";

    //setup visualizer and add query cloud 
    pcl::visualization::PCLVisualizer visu("KNN search");
    visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
    visu.addPointCloud<pcl::PointXYZ> (cloud, ColorHandler(cloud, 0.0 , 255.0, 0.0), "Query Cloud Cloud", viewport);
    visu.addText ("Query Cloud", 20, 30, 136.0/255.0, 58.0/255.0, 1, "Query Cloud", viewport); 
    visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, "Query Cloud", viewport);
    //visu.addCoordinateSystem (0.01, 0);

    //add matches to plot
    for(int i = 1; i < (k+1); ++i)
    {
        //shift viewpoint
        ++l;
        if (l >= x_s)
        {
          l = 0;
          m++;
        }

        //names and text labels
        std::string textString = viewName;
        std::string cloudname = viewName;
        textString.append(boost::lexical_cast<std::string>(i));
        cloudname.append(boost::lexical_cast<std::string>(i)).append("cloud");

        //color proportional to distance

        //add cloud
        visu.createViewPort (l * x_step, m * y_step, (l + 1) * x_step, (m + 1) * y_step, viewport);
        visu.addPointCloud<pcl::PointXYZ> (cloudList[i-1], ColorHandler(cloudList[i-1], 0.0 , 255.0, 0.0), cloudname, viewport);
        visu.addText (textString, 20, 30, 136.0/255.0, 58.0/255.0, 1, textString, viewport);
        visu.setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_FONT_SIZE, 18, textString, viewport);
    }
    visu.spin();

    return 0;
}
