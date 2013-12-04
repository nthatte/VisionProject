#ifndef VFHPOSEEST_H
#define VFHPOSEEST_H

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

class VFHPoseEstimator
{
private:
    typedef pcl::PointCloud<pcl::Normal> Normals;
    typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler;
    const int histLength;

    struct CloudInfo
    {
        float roll;
        float pitch; 
        float yaw; 
        boost::filesystem::path filePath;
        float hist[308];
    };

    /** \brief loads either a .pcd or .ply file into a pointcloud 
        \param cloud pointcloud to load data into
        \param path path to pointcloud file
    */
    bool loadPointCloud(const boost::filesystem::path &path, PointCloud &cloud);

    /** \brief Load the list of angles from FLANN list file
      * \param list of angles
      * \param filename the input file name
      */
    bool loadFLANNAngleData (std::vector<CloudInfo> &cloudInfoList, const std::string &filename);

    /** \brief loads angle data corresponding to a pointcloud
        \param path path to .txt file containing angle information
        \param cloudInfo stuct to load roll, pitch, yaw angles into
    */
    bool loadCloudAngleData(const boost::filesystem::path &path, CloudInfo &cloudInfo);

    /** \brief Search for the knn
      * \param index the tree
      * \param vfhs pointer to the query vfh feature
      * \param k the number of neighbors to search for
      * \param indices the resultant neighbor indices
      * \param distances the resultant neighbor distances
      */
    void nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances);

public:
    VFHPoseEstimator(): histLength(308) {}

    /** \brief Returns closest pose of closest cloud in training dataset to the query cloud
        \param cloud the query point cloud
        \param roll roll angle
        \param pitch pitch angle
        \param yaw yaw angle
        \param visMatch whether or not to visualze the closest match
    */
    bool getPose (const PointCloud::Ptr &cloud, float &roll, float &pitch, float &yaw, const bool visMatch);

    /** \brief Returns closest pose of closest cloud in training dataset to the query cloud
        \param dataDir boost path to directory with training data
    */
    bool trainClassifier(boost::filesystem::path &dataDir);
};
#endif
