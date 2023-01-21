/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2018, Locus Robotics
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
#include <fuse_publishers/pose_3d_publisher.h>

#include <fuse_core/async_publisher.h>
#include <fuse_core/graph.h>
#include <fuse_core/transaction.h>
#include <fuse_core/uuid.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/TransformStamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Register this publisher with ROS as a plugin.
PLUGINLIB_EXPORT_CLASS(fuse_publishers::Pose3DPublisher, fuse_core::Publisher);

// Some file-scope functions in an anonymous namespace
namespace
{
bool findPose(const fuse_core::Graph& graph, const ros::Time& stamp, const fuse_core::UUID& device_id,
              fuse_core::UUID& orientation_uuid, fuse_core::UUID& position_uuid, geometry_msgs::Pose& pose)
{
  try
  {
    orientation_uuid = fuse_variables::Orientation3DStamped(stamp, device_id).uuid();
    auto orientation_variable =
        dynamic_cast<const fuse_variables::Orientation3DStamped&>(graph.getVariable(orientation_uuid));
    position_uuid = fuse_variables::Position3DStamped(stamp, device_id).uuid();
    auto position_variable = dynamic_cast<const fuse_variables::Position3DStamped&>(graph.getVariable(position_uuid));
    pose.position.x = position_variable.x();
    pose.position.y = position_variable.y();
    pose.position.z = position_variable.z();
    pose.orientation.w = orientation_variable.w();
    pose.orientation.x = orientation_variable.x();
    pose.orientation.y = orientation_variable.y();
    pose.orientation.z = orientation_variable.z();
  }
  catch (const std::exception& e)
  {
    ROS_WARN_STREAM_THROTTLE(10.0, "Failed to find a pose at time " << stamp << ". Error" << e.what());
    return false;
  }
  catch (...)
  {
    ROS_WARN_STREAM_THROTTLE(10.0, "Failed to find a pose at time " << stamp << ". Error: unknown");
    return false;
  }
  return true;
}

}  // namespace

namespace fuse_publishers
{
Pose3DPublisher::Pose3DPublisher()
  : fuse_core::AsyncPublisher(1), device_id_(fuse_core::uuid::NIL), publish_to_tf_(false), use_tf_lookup_(false)
{
}

void Pose3DPublisher::onInit()
{
  // Read configuration from the parameter server
  private_node_handle_.param("base_frame", base_frame_, std::string("base_link"));
  private_node_handle_.param("map_frame", map_frame_, std::string("map"));
  private_node_handle_.param("odom_frame", odom_frame_, std::string("odom"));
  std::string device_str;
  if (private_node_handle_.getParam("device_id", device_str))
  {
    device_id_ = fuse_core::uuid::from_string(device_str);
  }
  else if (private_node_handle_.getParam("device_name", device_str))
  {
    device_id_ = fuse_core::uuid::generate(device_str);
  }
  private_node_handle_.param("publish_to_tf", publish_to_tf_, false);

  // Configure tf, if requested
  if (publish_to_tf_)
  {
    use_tf_lookup_ = (!odom_frame_.empty() && (odom_frame_ != base_frame_));
    if (use_tf_lookup_)
    {
      double tf_cache_time;
      double default_tf_cache_time = 10.0;
      private_node_handle_.param("tf_cache_time", tf_cache_time, default_tf_cache_time);
      if (tf_cache_time <= 0)
      {
        ROS_WARN_STREAM("The requested tf_cache_time is <= 0. Using the default value (" << default_tf_cache_time
                                                                                         << "s) instead.");
        tf_cache_time = default_tf_cache_time;
      }

      double tf_timeout;
      double default_tf_timeout = 0.1;
      private_node_handle_.param("tf_timeout", tf_timeout, default_tf_timeout);
      if (tf_timeout <= 0)
      {
        ROS_WARN_STREAM("The requested tf_timeout is <= 0. Using the default value (" << default_tf_timeout
                                                                                      << "s) instead.");
        tf_timeout = default_tf_timeout;
      }
      tf_timeout_ = ros::Duration(tf_timeout);

      tf_buffer_ = std::make_unique<tf2_ros::Buffer>(ros::Duration(tf_cache_time));
      tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, node_handle_);
    }

    double tf_publish_frequency;
    double default_tf_publish_frequency = 10.0;
    private_node_handle_.param("tf_publish_frequency", tf_publish_frequency, default_tf_publish_frequency);
    if (tf_publish_frequency <= 0)
    {
      ROS_WARN_STREAM("The requested tf_publish_frequency is <= 0. Using the default value ("
                      << default_tf_publish_frequency << "hz) instead.");
      tf_publish_frequency = default_tf_publish_frequency;
    }
    tf_publish_timer_ = private_node_handle_.createTimer(ros::Duration(1.0 / tf_publish_frequency),
                                                         &Pose3DPublisher::tfPublishTimerCallback, this, false, false);
  }

  // Advertise the topics
  pose_publisher_ = private_node_handle_.advertise<geometry_msgs::PoseStamped>("pose", 1);
  pose_with_covariance_publisher_ =
      private_node_handle_.advertise<geometry_msgs::PoseWithCovarianceStamped>("pose_with_covariance", 1);
}

void Pose3DPublisher::onStart()
{
  // Clear the transform
  tf_transform_ = geometry_msgs::TransformStamped();
  // Clear the synchronizer
  synchronizer_ = Synchronizer::make_unique(device_id_);
  // Start the tf timer
  if (publish_to_tf_)
  {
    tf_publish_timer_.start();
  }
}

void Pose3DPublisher::onStop()
{
  // Stop the tf timer
  if (publish_to_tf_)
  {
    tf_publish_timer_.stop();
  }
}

void Pose3DPublisher::notifyCallback(fuse_core::Transaction::ConstSharedPtr transaction,
                                     fuse_core::Graph::ConstSharedPtr graph)
{
  auto latest_stamp = synchronizer_->findLatestCommonStamp(*transaction, *graph);
  if (latest_stamp == Synchronizer::TIME_ZERO)
  {
    ROS_WARN_STREAM_THROTTLE(10.0, "Failed to find a matching set of stamped pose variables with device id '"
                                       << device_id_ << "'.");
    return;
  }
  // Get the pose values associated with the selected timestamp
  fuse_core::UUID orientation_uuid;
  fuse_core::UUID position_uuid;
  geometry_msgs::Pose pose;
  if (!findPose(*graph, latest_stamp, device_id_, orientation_uuid, position_uuid, pose))
  {
    return;
  }
  // Publish the various message types
  if (publish_to_tf_)
  {
    // Create a 3D ROS Transform message from the current 3D pose
    geometry_msgs::TransformStamped map_to_base;
    map_to_base.header.stamp = latest_stamp;
    map_to_base.header.frame_id = map_frame_;
    map_to_base.child_frame_id = base_frame_;
    map_to_base.transform.translation.x = pose.position.x;  // Transforms use Vector3 instead of Point (shakes fist)
    map_to_base.transform.translation.y = pose.position.y;
    map_to_base.transform.translation.z = pose.position.z;
    map_to_base.transform.rotation = pose.orientation;
    // If we are suppose to publish the map->odom frame instead, do that transformation now
    if (use_tf_lookup_)
    {
      // We need to lookup the base->odom frame first, so we can compute the map->odom transform from the
      // map->base transform
      try
      {
        auto base_to_odom = tf_buffer_->lookupTransform(base_frame_, odom_frame_, latest_stamp, tf_timeout_);
        geometry_msgs::TransformStamped map_to_odom;
        tf2::doTransform(base_to_odom, map_to_odom, map_to_base);
        map_to_odom.child_frame_id = odom_frame_;  // The child frame is not populated for some reason
        tf_transform_ = map_to_odom;
      }
      catch (const std::exception& e)
      {
        ROS_WARN_STREAM_THROTTLE(2.0, "Could not lookup the transform " << base_frame_ << "->" << odom_frame_
                                                                        << ". Error: " << e.what());
      }
    }
    else
    {
      // Simple. No intermediate frame. Just use the optimized map->base transform.
      tf_transform_ = map_to_base;
    }
  }
  if (pose_publisher_.getNumSubscribers() > 0)
  {
    geometry_msgs::PoseStamped msg;
    msg.header.stamp = latest_stamp;
    msg.header.frame_id = map_frame_;
    msg.pose = pose;
    pose_publisher_.publish(msg);
  }
  if (pose_with_covariance_publisher_.getNumSubscribers() > 0)
  {
    // Get the covariance from the graph
    std::vector<std::pair<fuse_core::UUID, fuse_core::UUID>> requests;
    requests.emplace_back(position_uuid, position_uuid);
    requests.emplace_back(position_uuid, orientation_uuid);
    requests.emplace_back(orientation_uuid, orientation_uuid);
    std::vector<std::vector<double>> covariance_blocks;
    graph->getCovariance(requests, covariance_blocks);
    geometry_msgs::PoseWithCovarianceStamped msg;
    msg.header.stamp = latest_stamp;
    msg.header.frame_id = map_frame_;
    msg.pose.pose = pose;
    msg.pose.covariance[0] = covariance_blocks[0][0];
    msg.pose.covariance[1] = covariance_blocks[0][1];
    msg.pose.covariance[2] = covariance_blocks[0][2];
    msg.pose.covariance[3] = covariance_blocks[0][3];
    msg.pose.covariance[4] = covariance_blocks[0][4];
    msg.pose.covariance[5] = covariance_blocks[0][5];
    msg.pose.covariance[6] = covariance_blocks[1][0];
    msg.pose.covariance[7] = covariance_blocks[1][1];
    msg.pose.covariance[8] = covariance_blocks[1][2];
    msg.pose.covariance[9] = covariance_blocks[1][3];
    msg.pose.covariance[10] = covariance_blocks[1][4];
    msg.pose.covariance[11] = covariance_blocks[1][5];
    msg.pose.covariance[12] = covariance_blocks[2][0];
    msg.pose.covariance[13] = covariance_blocks[2][1];
    msg.pose.covariance[14] = covariance_blocks[2][2];
    msg.pose.covariance[15] = covariance_blocks[2][3];
    msg.pose.covariance[16] = covariance_blocks[2][4];
    msg.pose.covariance[17] = covariance_blocks[2][5];
    msg.pose.covariance[18] = covariance_blocks[3][0];
    msg.pose.covariance[19] = covariance_blocks[3][1];
    msg.pose.covariance[20] = covariance_blocks[3][2];
    msg.pose.covariance[21] = covariance_blocks[3][3];
    msg.pose.covariance[22] = covariance_blocks[3][4];
    msg.pose.covariance[23] = covariance_blocks[3][5];
    msg.pose.covariance[24] = covariance_blocks[4][0];
    msg.pose.covariance[25] = covariance_blocks[4][1];
    msg.pose.covariance[26] = covariance_blocks[4][2];
    msg.pose.covariance[27] = covariance_blocks[4][3];
    msg.pose.covariance[28] = covariance_blocks[4][4];
    msg.pose.covariance[29] = covariance_blocks[4][5];
    msg.pose.covariance[30] = covariance_blocks[5][0];
    msg.pose.covariance[31] = covariance_blocks[5][1];
    msg.pose.covariance[32] = covariance_blocks[5][2];
    msg.pose.covariance[33] = covariance_blocks[5][3];
    msg.pose.covariance[34] = covariance_blocks[5][4];
    msg.pose.covariance[35] = covariance_blocks[5][5];

    pose_with_covariance_publisher_.publish(msg);
  }
}

void Pose3DPublisher::tfPublishTimerCallback(const ros::TimerEvent& event)
{
  // The tf_transform_ is updated in a separate thread, so we must guard the read/write operations.
  // Only publish if the tf transform is valid
  if (tf_transform_.header.stamp != Synchronizer::TIME_ZERO)
  {
    // Update the timestamp of the transform so the tf tree will continue to be valid
    tf_transform_.header.stamp = event.current_real;
    tf_publisher_.sendTransform(tf_transform_);
  }
}

}  // namespace fuse_publishers
