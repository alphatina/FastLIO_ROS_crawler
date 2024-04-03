// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN];
int s_plot1[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN];
double s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN], s_plot12[MAXN], s_plot13[MAXN];
double match_time = 0, solve_ESKF_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_num = 0;
bool   runtime_pos_log = false, pcd_save_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable cv_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lidar_topic, lidar_topic2, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.001, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effect_feat_num = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
vector<double>       lidar2baseT(3, 0.0);
vector<double>       lidar2baseR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);
V3D IMU_T_wrt_base(Zero3d);
M3D IMU_R_wrt_base(Eye3d);
vect3 base_T_wrt_IMU(Zero3d);
M3D base_R_wrt_IMU(Eye3d);
V3D Lidar_T_wrt_base(Zero3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lidar;
V3D pos_imu_in_base;
V3D pos_base;
V3D rot_angle_degree;
M3D rot_matrix;
M3D rot_base;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
  flg_exit = true;
  ROS_WARN("catch sig %d", sig);
  cv_buffer.notify_one();
}

inline void dump_lio_state_to_log(FILE *fp_state)  
{
  fprintf(fp_state, "%.3f ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp_state, "%.3f %.3f %.3f ", rot_angle_degree(2), rot_angle_degree(1), rot_angle_degree(0));  // Angle
  fprintf(fp_state, "%.3f %.3f %.3f ", pos_base(0), pos_base(1), pos_base(2)); // Pos  
  fprintf(fp_state, "%.3f %.3f %.3f ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
  fprintf(fp_state, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
  fprintf(fp_state, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
  fprintf(fp_state, "%lf %lf %lf", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
  fprintf(fp_state, "\r\n");  
  fflush(fp_state);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  // V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
  V3D p_global(rot_matrix * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + pos_imu_in_base);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_num = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiDAR = pos_lidar;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiDAR(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiDAR(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiDAR(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiDAR(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) {
      kdtree_delete_num = ikdtree.Delete_Point_Boxes(cub_needrm);
    }
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
  // double preprocess_start_time = omp_get_wtime();
  if (msg->header.stamp.toSec() < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = msg->header.stamp.toSec();

  PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  mtx_buffer.lock();
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(msg->header.stamp.toSec());
  mtx_buffer.unlock();
  cv_buffer.notify_one();
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
  // double preprocess_start_time = omp_get_wtime();
  if (msg->header.stamp.toSec() < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = msg->header.stamp.toSec();
  
  PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  std::lock_guard<std::mutex> lock(mtx_buffer);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);
  cv_buffer.notify_one();
}

void livox_pcl_cbk2(const livox_ros_driver::CustomMsg::ConstPtr &msg1, const livox_ros_driver::CustomMsg::ConstPtr &msg2) 
{
  // double preprocess_start_time = omp_get_wtime();
  p_pre->time_diff_lidar12 = msg1->header.stamp.toSec() - msg2->header.stamp.toSec();
  if (p_pre->time_diff_lidar12 > 0) {
    last_timestamp_lidar = msg2->header.stamp.toSec();
  } else {
    last_timestamp_lidar = msg1->header.stamp.toSec();
  }
  
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process2(msg1, msg2, ptr);

  std::lock_guard<std::mutex> lock(mtx_buffer);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);
  cv_buffer.notify_one();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
  // sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  double timestamp = msg_in->header.stamp.toSec();
  if (timestamp < last_timestamp_imu)
  {
    ROS_WARN("imu loop back, clear buffer");
    imu_buffer.clear();
  }
  last_timestamp_imu = timestamp;

  std::lock_guard<std::mutex> lock(mtx_buffer);
  imu_buffer.push_back(msg_in);
  cv_buffer.notify_one();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
  if (lidar_buffer.empty() || imu_buffer.empty()) {
    return false;
  }

  /*** push a lidar scan ***/
  if(!lidar_pushed)
  {
    meas.lidar = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();
    if (meas.lidar->points.size() <= 1) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
      ROS_WARN("Too few input point cloud!\n");
    }
    else if (meas.lidar->points.back().curvature < 0.5 * lidar_mean_scantime) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    }
    else {
      scan_num ++;
      lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature;
      lidar_mean_scantime += (meas.lidar->points.back().curvature - lidar_mean_scantime) / scan_num;
    }

    meas.lidar_end_time = lidar_end_time;

    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time)
    return false;

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = imu_buffer.front()->header.stamp.toSec();
  meas.imu.clear();
  while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
  {
    imu_time = imu_buffer.front()->header.stamp.toSec();
    if(imu_time > lidar_end_time) break;
    meas.imu.push_back(imu_buffer.front());
    imu_buffer.pop_front();
  }
  lidar_buffer.pop_front();
  time_buffer.pop_front();

  lidar_pushed = false;
  return true;

}

int process_increments = 0;
void map_incremental()
{
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < feats_down_size; i++) {
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
  }

  for (int i = 0; i < feats_down_size; i++)
  {
    /* decide if need add to map */
    if (!Nearest_Points[i].empty() && flg_EKF_inited)
    {
      const PointVector &points_near = Nearest_Points[i];
      bool need_add = true;
      BoxPointType Box_of_Point;
      PointType downsample_result, mid_point; 
      mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
      mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
      float dist  = calc_dist(feats_down_world->points[i],mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
        PointNoNeedDownsample.push_back(feats_down_world->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
      {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist)
        {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
    }
    else
    {
      PointToAdd.push_back(feats_down_world->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  add_point_size = ikdtree.Add_Points(PointToAdd, true);
  ikdtree.Add_Points(PointNoNeedDownsample, false); 
  add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time = omp_get_wtime() - st_time;
}

// PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
// PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
std::vector<PointCloudXYZI::Ptr> pcl_wait_save;   /// 存储等待保存的点云
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
  PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < size; i++) {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }

  if(scan_pub_en)
  {
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "map";
    pubLaserCloudFull.publish(laserCloudmsg);
  }

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en)
  {
    pcl_wait_save.emplace_back(laserCloudWorld);

    // static int scan_wait_num = 0;
    // scan_wait_num ++;
    // if (pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval && pcl_wait_save->size() > 0)
    // {
    //   pcd_index ++;
    //   string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
    //   pcl::PCDWriter pcd_writer;
    //   cout << "current scan saved to /PCD/" << all_points_dir << endl;
    //   pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    //   pcl_wait_save->clear();
    //   scan_wait_num = 0;
    // }
  }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < size; i++) {
    RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudmsg.header.frame_id = "baselink";
  pubLaserCloudFull_body.publish(laserCloudmsg);
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  #ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < effect_feat_num; i++) {
    RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudFullRes3.header.frame_id = "map";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
  sensor_msgs::PointCloud2 laserCloudMap;
  pcl::toROSMsg(*featsFromMap, laserCloudMap);
  laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
  laserCloudMap.header.frame_id = "map";
  pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
  // out.pose.position.x = state_point.pos(0);
  // out.pose.position.y = state_point.pos(1);
  // out.pose.position.z = state_point.pos(2);
  out.pose.position.x = pos_base(0);
  out.pose.position.y = pos_base(1);
  out.pose.position.z = pos_base(2);
  out.pose.orientation.x = geoQuat.x;
  out.pose.orientation.y = geoQuat.y;
  out.pose.orientation.z = geoQuat.z;
  out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "map";
  odomAftMapped.child_frame_id = "baselink";
  odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
  set_posestamp(odomAftMapped.pose);
  pubOdomAftMapped.publish(odomAftMapped);
  auto P = kf.get_P();
  for (int i = 0; i < 6; i ++)
  {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
    odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
    odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
    odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
    odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
    odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
  }

  static tf::TransformBroadcaster br;
  tf::Transform                   transform;
  tf::Quaternion                  q;
  transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                  odomAftMapped.pose.pose.position.y, \
                                  odomAftMapped.pose.pose.position.z));
  q.setW(odomAftMapped.pose.pose.orientation.w);
  q.setX(odomAftMapped.pose.pose.orientation.x);
  q.setY(odomAftMapped.pose.pose.orientation.y);
  q.setZ(odomAftMapped.pose.pose.orientation.z);
  transform.setRotation( q );
  br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "map", "baselink" ) );
}

void publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose);
  msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
  msg_body_pose.header.frame_id = "map";

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) 
  {
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
  }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effect_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
      if (point_selected_surf[i])
      {
        laserCloudOri->points[effect_feat_num] = feats_down_body->points[i];
        corr_normvect->points[effect_feat_num] = normvec->points[i];
        total_residual += res_last[i];
        effect_feat_num ++;
      }
    }

    if (effect_feat_num < 1)
    {
      ekfom_data.valid = false;
      ROS_WARN("No Effective Points! \n");
      return;
    }

    res_mean_last = total_residual / effect_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effect_feat_num, 12); //23
    ekfom_data.h.resize(effect_feat_num);

    for (int i = 0; i < effect_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_ESKF_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lidar_topic",lidar_topic,"/livox/lidar");
    nh.param<string>("common/lidar_topic2",lidar_topic2,"/livox/lidar2");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<int>("common/imu_init_count", p_imu->imu_init_count, 100);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("mapping/lidar2base_T", lidar2baseT, vector<double>());
    nh.param<vector<double>>("mapping/lidar2base_R", lidar2baseR, vector<double>());
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    p_pre->blind2 = p_pre->blind * p_pre->blind;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = "map";
    
    /*** variables definition ***/
    int frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    Lidar_T_wrt_base<<VEC_FROM_ARRAY(lidar2baseT);
    IMU_R_wrt_base<<MAT_FROM_ARRAY(lidar2baseR);
    base_R_wrt_IMU = IMU_R_wrt_base.transpose();
    IMU_T_wrt_base = Lidar_T_wrt_base - IMU_R_wrt_base * Lidar_T_wrt_IMU;
    base_T_wrt_IMU = Lidar_T_wrt_IMU - base_R_wrt_IMU * Lidar_T_wrt_base;
    
    std::cout << "Lidar pos in IMU: " << Lidar_T_wrt_IMU.transpose() << ", Lidar2IMU R: " <<endl<< Lidar_R_wrt_IMU <<endl;
    std::cout << "IMU pos in base: " << IMU_T_wrt_base.transpose() << ", IMU2base R: " <<endl<< IMU_R_wrt_base <<endl;

    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp_state;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp_state = fopen(pos_log_dir.c_str(),"w");
    if (fp_state == NULL)
      ROS_ERROR("/Log/pos_log.txt doesn't exist");

    /*** ROS subscribe initialization ***/
    // ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lidar_topic, 10, livox_pcl_cbk) : nh.subscribe(lidar_topic, 10, standard_pcl_cbk);
    
    message_filters::Subscriber<livox_ros_driver::CustomMsg> lidar_sub1(nh, lidar_topic, 10);
    message_filters::Subscriber<livox_ros_driver::CustomMsg> lidar_sub2(nh, lidar_topic2, 10);
    typedef message_filters::sync_policies::ApproximateTime<livox_ros_driver::CustomMsg, livox_ros_driver::CustomMsg> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), lidar_sub1, lidar_sub2);
    sync.registerCallback(boost::bind(&livox_pcl_cbk2, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 10);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 10);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 10);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 10);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 10);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 10);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
      if (flg_exit) break;
      ros::spinOnce();
      if(sync_packages(Measures)) 
      {
        if (flg_first_scan)
        {
          first_lidar_time = Measures.lidar_beg_time;
          p_imu->first_lidar_time = first_lidar_time;
          flg_first_scan = false;
          std::cout << "first scan init at " << fixed << setprecision(3) << first_lidar_time << "s" << endl;
          continue;
        }

        double t0,t1,t2,t3,t4, match_start, solve_start, svd_time;

        match_time = 0;
        kdtree_search_time = 0.0;
        solve_ESKF_time = 0;
        svd_time   = 0;
        t0 = omp_get_wtime();

        p_imu->Process(Measures, kf, feats_undistort);
        state_point = kf.get_x();
        
        pos_lidar = state_point.pos + state_point.rot * state_point.offset_T_L_I;

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
          ROS_WARN("No point, skip this scan at %.3f s!\n", lidar_end_time-first_lidar_time);
          continue;
        }

        flg_EKF_inited = (Measures.lidar_beg_time-first_lidar_time) < INIT_TIME ? false : true;
        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        feats_down_size = feats_down_body->points.size();
        /*** initialize the map kdtree ***/
        if(ikdtree.Root_Node == nullptr)
        {
          if(feats_down_size > 5)
          {
            ikdtree.set_downsample_param(filter_size_map_min);
            feats_down_world->resize(feats_down_size);
            #ifdef MP_EN
              omp_set_num_threads(MP_PROC_NUM);
              #pragma omp parallel for
            #endif
            for(int i = 0; i < feats_down_size; i++) {
              pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Build(feats_down_world->points);
            std::cout << "ikdtree init at " << fixed << setprecision(3) << lidar_end_time-first_lidar_time << "s, imu pose: " << state_point.pos << "m" << endl << endl;
          }
          continue;
        }
        int featsFromMapNum = ikdtree.validnum();
        // kdtree_size_st = ikdtree.size();
          
        /*** ICP and iterated Kalman filter update ***/
        if (feats_down_size < 5)
        {
          ROS_WARN("No point, skip this scan at %.3f s!\n", lidar_end_time-first_lidar_time);
          continue;
        }
        
        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);

        // if(0) // If you need to see map point, change to "if(1)"
        // {
        //   PointVector ().swap(ikdtree.PCL_Storage);
        //   ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        //   featsFromMap->clear();
        //   featsFromMap->points = ikdtree.PCL_Storage;
        // }

        pointSearchInd_surf.resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        int  rematch_num = 0;
        bool nearest_search_en = true; //

        t1 = omp_get_wtime();
          
        /*** iterated state estimation ***/
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

        state_point = kf.get_x();
        pos_lidar = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        
        rot_matrix = IMU_R_wrt_base * state_point.rot.toRotationMatrix();
        pos_imu_in_base = IMU_R_wrt_base * state_point.pos + IMU_T_wrt_base;
        pos_base = rot_matrix * base_T_wrt_IMU + pos_imu_in_base;
        rot_base = rot_matrix * base_R_wrt_IMU;
        rot_angle_degree = rot_base.eulerAngles(2,1,0)*180/M_PI;
        Eigen::Quaterniond rot_base_q(rot_base);
        geoQuat.x = rot_base_q.x();
        geoQuat.y = rot_base_q.y();
        geoQuat.z = rot_base_q.z();
        geoQuat.w = rot_base_q.w();

        // geoQuat.x = state_point.rot.coeffs()[0];
        // geoQuat.y = state_point.rot.coeffs()[1];
        // geoQuat.z = state_point.rot.coeffs()[2];
        // geoQuat.w = state_point.rot.coeffs()[3];

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped);

        /*** add the feature points to map kdtree ***/
        t2 = omp_get_wtime();
        map_incremental();
        t3 = omp_get_wtime();
        
        /******* Publish points *******/
        if (path_en)                         publish_path(pubPath);
        if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
        // publish_effect_world(pubLaserCloudEffect);
        // publish_map(pubLaserCloudMap);

        t4 = omp_get_wtime();
        /*** Debug variables ***/
        if (runtime_pos_log)
        {
          // double coef_add = 1.0 / (frame_num + 1.0);
          kdtree_size_end = ikdtree.size();
          aver_time_consu = (t4-t0)*1e3;
          aver_time_solve = (solve_H_time+solve_ESKF_time)*1e3;
          T1[frame_num] = lidar_end_time - first_lidar_time;
          s_plot1[frame_num] = feats_undistort->points.size();
          s_plot2[frame_num] = feats_down_size;
          s_plot3[frame_num] = kdtree_size_end;
          s_plot4[frame_num] = featsFromMapNum;
          s_plot5[frame_num] = effect_feat_num;
          s_plot6[frame_num] = add_point_size;
          s_plot7[frame_num] = kdtree_delete_num;
          s_plot8[frame_num] = res_mean_last;
          s_plot9[frame_num] = aver_time_consu;   //// 整个流程的总时间
          s_plot10[frame_num] = (t1-t0)*1e3;
          s_plot11[frame_num] = (t2-t1)*1e3;
          s_plot12[frame_num] = (t3-t2)*1e3;
          s_plot13[frame_num] = (t4-t3)*1e3;
          frame_num++;
          

          dump_lio_state_to_log(fp_state);
          // dump_lio_cov_to_log(fp_ESKF_cov);
          if (frame_num % 10 == 1)
          {
            cout << "Lidar time: " << lidar_end_time-first_lidar_time << "s, ";
            // cout << "scan origin: " << feats_undistort->points.size() << ", match: "<< effect_feat_num << ", res mean: " << res_mean_last << endl;
            cout << "pose: " << pos_base.transpose()<< " m, " << rot_angle_degree.transpose() <<  "°" << endl;
            cout << "Total cost time: " << aver_time_consu << ", ESKF: " << s_plot11[frame_num-1] << " ms" << endl;
          //   // cout << "IMU compensate: " << (t1-t0)*1e3 << ", " 
          //   //      << "map adjust: " << (t2-t1)*1e3 << ", "
          //   //      << "Downsample + Lidar2World: " << (t3-t2)*1e3 << " ms" << endl
          //   //      << "ESKF: " << (t4-t3)*1e3 
          //   //      << ", match " << match_time*1e3 
          //   //      << ", solve: " << aver_time_solve << " ms" << endl
          //   //      << "Map increment: " << (t5-t4)*1e3 << " ms" << endl;
          //   // ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            std::cout << endl;
          }
        }
      }
      status = ros::ok();
      rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    scan_num = pcl_wait_save.size();
    if (pcd_save_en && scan_num > 0)
    {
      string map_path = root_dir + "PCD/scans.pcd";
      cout << "Map saved in " << map_path << endl; 
      PointCloudXYZI::Ptr pointcloud_map(new PointCloudXYZI());
      int map_point_num = 0;
      for (int i=0; i<scan_num; ++i) {
        map_point_num += pcl_wait_save[i]->points.size();
        *pointcloud_map += *pcl_wait_save[i];
      }
      cout << "scan num is " << scan_num << ", point num is " << map_point_num << endl;
      pcl::PCDWriter pcd_writer;
      pcd_writer.writeBinary(map_path, *pointcloud_map);
    }

    if (runtime_pos_log)
    {
      FILE *fp2;
      string log_path = root_dir + "/Log/fastlio_time_log.csv";
      fp2 = fopen(log_path.c_str(),"w");
      if (fp2 == NULL) {
        ROS_ERROR("/Log/fastlio_time_log.csv doesn't exist");
      }
      fprintf(fp2, "lidar time, scan point num, downsample scan point num, ikd-tree point num, ikd-tree effect point num, match point num, add point num, delete point num, residual, total time, preprocessing time, ESKF time, map increment time, publish time \r\n");
      for (int i = 0;i<frame_num; i++){
        fprintf(fp2, "%.3f, %d, %d, %d, %d, %d, %d, %d, %.3f, %.2f, %.2f, %.2f, %.2f, %.2f \r\n",
                T1[i], s_plot1[i], s_plot2[i], s_plot3[i], s_plot4[i], s_plot5[i], s_plot6[i], s_plot7[i], s_plot8[i], s_plot9[i], s_plot10[i], s_plot11[i], s_plot12[i], s_plot13[i]);
      }
      fflush(fp2);
      fclose(fp2);
    }

    return 0;
}
