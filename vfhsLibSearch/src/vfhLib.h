#ifndef VFHLIB_H
#define VFHLIB_H

typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandler;
const int histLength = 308;

/** \brief loads either a .pcd or .ply file into a pointcloud 
    \param cloud pointcloud to load data into
    \param path path to pointcloud file
*/
bool loadPointCloud(const boost::filesystem::path &path, PointCloud &cloud);

/** \brief Returns closest pose of closest cloud in training dataset to the query cloud
    \param cloud the query point cloud
    \param roll roll angle
    \param pitch pitch angle
    \param yaw yaw angle
    \param visMatch whether or not to visualze the closest match
*/
bool getPose (const PointCloud &cloud, float &roll, float &pitch, float &yaw, const bool visMatch);

#endif
