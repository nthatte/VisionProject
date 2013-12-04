#include "VFHPoseEstimation.h"

/** \brief loads either a .pcd or .ply file into a pointcloud 
    \param cloud pointcloud to load data into
    \param path path to pointcloud file
*/
bool VFHPoseEstimator::loadPointCloud(const boost::filesystem::path &path, PointCloud &cloud)
{
    std::cout << "Loading: " << path.filename() << std::endl;
    //read pcd file
    pcl::PCDReader reader;
    if( reader.read(path.native(), cloud) == -1)
    {
        PCL_ERROR("Could not read .pcd file\n");
        return false;
    }
    return true;
}

/** \brief Load the list of angles from FLANN list file
  * \param list of angles
  * \param filename the input file name
  */
bool VFHPoseEstimator::loadFLANNAngleData (std::vector<VFHPoseEstimator::CloudInfo> &cloudInfoList, const std::string &filename)
{
    ifstream fs;
    fs.open (filename.c_str ());
    if (!fs.is_open () || fs.fail ())
        return (false);

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

        //read yaw
        std::getline (fs, line, ' ');
        if (line.empty ())
            continue;
        cloudinfo.yaw = boost::lexical_cast<float>(line);

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

/** \brief loads either angle data corresponding  
    \param path path to .txt file containing angle information
    \param cloudInfo stuct to load theta and phi angles into
*/
bool VFHPoseEstimator::loadCloudAngleData(const boost::filesystem::path &path, CloudInfo &cloudInfo)
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
    cloudInfo.roll = boost::lexical_cast<float>(angle);
    std::getline (fs, angle);
    cloudInfo.pitch = boost::lexical_cast<float>(angle);
    cloudInfo.yaw = 0;
    fs.close ();

    //save filename
    cloudInfo.filePath = path;
    cloudInfo.filePath.replace_extension(".pcd");
    return true;
}

/** \brief Search for the closest k neighbors
  * \param index the tree
  * \param vfhs pointer to the query vfh feature
  * \param k the number of neighbors to search for
  * \param indices the resultant neighbor indices
  * \param distances the resultant neighbor distances
  */
void VFHPoseEstimator::nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
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

bool VFHPoseEstimator::getPose (const PointCloud::Ptr &cloud, float &roll, float &pitch, float &yaw, const bool visMatch)
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
    if(!loadFLANNAngleData(cloudInfoList, anglesFileName))
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

bool VFHPoseEstimator::trainClassifier(boost::filesystem::path &dataDir)
{
    //loop over all pcd files in the data directry and calculate vfh features
    PointCloud::Ptr cloud (new PointCloud);
    Eigen::Matrix<float,4,1> centroid;
    std::list<CloudInfo> training; //training data list
    boost::filesystem::directory_iterator dirItr(dataDir), dirEnd;
    boost::filesystem::path angleDataPath;

    for(dirItr; dirItr != dirEnd; ++dirItr)
    {
        //skip txt and other files
        if(dirItr->path().extension().native().compare(".pcd") != 0)
            continue;

        //load point cloud
        if(!loadPointCloud(dirItr->path(), *cloud))
            return false;
        
        //load angle data from txt file
        angleDataPath = dirItr->path();
        angleDataPath.replace_extension(".txt");
        CloudInfo cloudInfo;
        if(!loadCloudAngleData(angleDataPath, cloudInfo))
            return false;

        //setup normal estimation class
        Normals::Ptr normals (new Normals);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr normTree (new pcl::search::KdTree<pcl::PointXYZ>);
        normEst.setSearchMethod(normTree);
        normEst.setRadiusSearch(0.005);

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
            cloudInfo.hist[i] = vfhs->points[0].histogram[i];
        }
        training.push_front(cloudInfo);
    }

    //convert training data to FLANN format
    flann::Matrix<float> data (new float[training.size() * histLength], training.size(), histLength);
    size_t i = 0;
    std::list<CloudInfo>::iterator it;
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
        fs << it->roll << " " << it->pitch << " " << it->yaw << " " << it->filePath.native() << "\n";
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

    return true;
}
