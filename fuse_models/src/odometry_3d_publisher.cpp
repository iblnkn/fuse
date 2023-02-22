/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2019, Locus Robotics
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
#include <fuse_core/async_publisher.h>
#include <fuse_core/eigen.h>
#include <fuse_core/uuid.h>
#include <fuse_models/common/sensor_proc.h>
#include <fuse_models/odometry_3d_publisher.h>
#include <fuse_models/skid_steer_3d_predict.h>
#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <pluginlib/class_list_macros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Register this publisher with ROS as a plugin.
PLUGINLIB_EXPORT_CLASS(fuse_models::Odometry3DPublisher, fuse_core::Publisher)

namespace fuse_models
{
Odometry3DPublisher::Odometry3DPublisher()
  : fuse_core::AsyncPublisher(1)
  , device_id_(fuse_core::uuid::NIL)
  , latest_stamp_(Synchronizer::TIME_ZERO)
  , latest_covariance_stamp_(Synchronizer::TIME_ZERO)
  , publish_timer_spinner_(1, &publish_timer_callback_queue_)
{
}

void Odometry3DPublisher::onInit()
{
  // Read settings from the parameter server
  device_id_ = fuse_variables::loadDeviceId(private_node_handle_);

  // Load parameters from the ROS parameter server
  params_.loadFromROS(private_node_handle_);

  // If the transform is not inverted and the world and map frame IDs are the
  // same, create a TransformListener
  if (!params_.invert_tf && params_.world_frame_id == params_.map_frame_id)
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(params_.tf_cache_time);
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_, node_handle_);
  }

  // Advertise a topic to publish odometry messages
  odom_pub_ = node_handle_.advertise<nav_msgs::Odometry>(ros::names::resolve(params_.topic), params_.queue_size);

  // Advertise a topic to publish acceleration messages
  acceleration_pub_ = node_handle_.advertise<geometry_msgs::AccelWithCovarianceStamped>(
      ros::names::resolve(params_.acceleration_topic), params_.queue_size);

  // Set up a timer to periodically publish odometry messages
  publish_timer_node_handle_.setCallbackQueue(&publish_timer_callback_queue_);

  publish_timer_ = publish_timer_node_handle_.createTimer(
      ros::Duration(1.0 / params_.publish_frequency), &Odometry3DPublisher::publishTimerCallback, this, false, false);

  // Start spinning the callback queue for the timer
  publish_timer_spinner_.start();
}

void Odometry3DPublisher::notifyCallback(
    fuse_core::Transaction::ConstSharedPtr transaction,  // the new transaction that has just been added to the
                                                         // graph
    fuse_core::Graph::ConstSharedPtr graph)
{  // the current graph of poses
  // Find the most recent common timestamp
  const auto latest_stamp =
      synchronizer_.findLatestCommonStamp(*transaction,
                                          *graph);  // Find the timestamp when the robot state is most recent.

  if (latest_stamp == Synchronizer::TIME_ZERO)
  {  // if the time is not found
    {
      std::lock_guard<std::mutex> lock(mutex_);  // acquire the lock
      latest_stamp_ = latest_stamp;              // Set the latest timestamp to TIME_ZERO
    }

    ROS_WARN_STREAM_THROTTLE(10.0,  // The message is throttled, meaning the message is printed every
                                    // 10 seconds
                             "Failed to find a matching set of state variables with device id '"
                                 << device_id_ << "'.");  // print warning message with robot device id
    return;
  }

  // Get the pose values associated with the selected timestamp
  fuse_core::UUID position_uuid;
  fuse_core::UUID orientation_uuid;
  fuse_core::UUID velocity_linear_uuid;
  fuse_core::UUID velocity_angular_uuid;
  fuse_core::UUID acceleration_linear_uuid;
  fuse_core::UUID acceleration_angular_uuid;

  nav_msgs::Odometry odom_output;                                 // message to be published
  geometry_msgs::AccelWithCovarianceStamped acceleration_output;  // message to be published
  if (!getState(*graph, latest_stamp, device_id_,
                position_uuid,  // get the most recent state estimate at timestamp
                orientation_uuid, velocity_linear_uuid, velocity_angular_uuid, acceleration_linear_uuid,
                acceleration_angular_uuid, odom_output, acceleration_output))
  {                                            // returns false if it is not possible to get
                                               // the state estimate
    std::lock_guard<std::mutex> lock(mutex_);  // acquire the lock
    latest_stamp_ = latest_stamp;              // Set the latest timestamp to the current timestamp
    return;
  }

  // set the header of the odometry message with the world frame id
  odom_output.header.frame_id = params_.world_frame_id;

  // set the header of the odometry message with the most recent timestamp
  odom_output.header.stamp = latest_stamp;

  // set the child frame id of the odometry message
  odom_output.child_frame_id = params_.base_link_output_frame_id;
  // set the header of the acceleration message with the base link output frame
  // id
  acceleration_output.header.frame_id = params_.base_link_output_frame_id;
  acceleration_output.header.stamp = latest_stamp;  // set the header of the acceleration message with the most
                                                    // recent timestamp

  // Don't waste CPU computing the covariance if nobody is listening
  ros::Time latest_covariance_stamp = latest_covariance_stamp_;  // set the latest_covariance_stamp to the
                                                                 // current value of the class member variable
  bool latest_covariance_valid = latest_covariance_valid_;       // set the latest_covariance_valid flag to the
                                                                 // current value of the class member variable

  // Compute the covariance only if there is a subscriber for odometry or
  // acceleration or the covariance throttle period has expired
  if (odom_pub_.getNumSubscribers() > 0 || acceleration_pub_.getNumSubscribers() > 0)
  {
    // Throttle covariance computation
    if (params_.covariance_throttle_period.isZero() ||
        latest_stamp - latest_covariance_stamp > params_.covariance_throttle_period)
    {
      latest_covariance_stamp = latest_stamp;

      try
      {
        std::vector<std::pair<fuse_core::UUID, fuse_core::UUID>> covariance_requests;
        covariance_requests.emplace_back(position_uuid, position_uuid);
        covariance_requests.emplace_back(position_uuid, orientation_uuid);
        covariance_requests.emplace_back(orientation_uuid, orientation_uuid);
        covariance_requests.emplace_back(velocity_linear_uuid, velocity_linear_uuid);
        covariance_requests.emplace_back(velocity_linear_uuid, velocity_angular_uuid);
        covariance_requests.emplace_back(velocity_angular_uuid, velocity_angular_uuid);
        covariance_requests.emplace_back(acceleration_linear_uuid, acceleration_linear_uuid);

        std::vector<std::vector<double>> covariance_matrices;
        graph->getCovariance(covariance_requests, covariance_matrices, params_.covariance_options);

        odom_output.pose.covariance[0] = covariance_matrices[0][0];
        odom_output.pose.covariance[1] = covariance_matrices[0][1];
        odom_output.pose.covariance[2] = covariance_matrices[0][2];

        odom_output.pose.covariance[3] = covariance_matrices[1][0];
        odom_output.pose.covariance[4] = covariance_matrices[1][1];
        odom_output.pose.covariance[5] = covariance_matrices[1][2];

        odom_output.pose.covariance[6] = covariance_matrices[0][3];
        odom_output.pose.covariance[7] = covariance_matrices[0][4];
        odom_output.pose.covariance[7] = covariance_matrices[0][5];

        odom_output.pose.covariance[9] = covariance_matrices[1][3];
        odom_output.pose.covariance[10] = covariance_matrices[1][4];
        odom_output.pose.covariance[11] = covariance_matrices[1][5];

        odom_output.twist.covariance[0] = covariance_matrices[3][0];
        odom_output.twist.covariance[1] = covariance_matrices[3][1];
        odom_output.twist.covariance[5] = covariance_matrices[4][0];
        odom_output.twist.covariance[6] = covariance_matrices[3][2];
        odom_output.twist.covariance[7] = covariance_matrices[3][3];
        odom_output.twist.covariance[11] = covariance_matrices[4][1];
        odom_output.twist.covariance[30] = covariance_matrices[4][0];
        odom_output.twist.covariance[31] = covariance_matrices[4][1];
        odom_output.twist.covariance[35] = covariance_matrices[5][0];

        acceleration_output.accel.covariance[0] = covariance_matrices[6][0];
        acceleration_output.accel.covariance[1] = covariance_matrices[6][1];
        acceleration_output.accel.covariance[6] = covariance_matrices[6][2];
        acceleration_output.accel.covariance[7] = covariance_matrices[6][3];

        latest_covariance_valid = true;
      }
      catch (const std::exception& e)
      {
        ROS_WARN_STREAM("An error occurred computing the covariance information for " << latest_stamp
                                                                                      << ". "
                                                                                         "The covariance will be set "
                                                                                         "to zero.\n"
                                                                                      << e.what());
        std::fill(odom_output.pose.covariance.begin(), odom_output.pose.covariance.end(), 0.0);
        std::fill(odom_output.twist.covariance.begin(), odom_output.twist.covariance.end(), 0.0);
        std::fill(acceleration_output.accel.covariance.begin(), acceleration_output.accel.covariance.end(), 0.0);

        latest_covariance_valid = false;
      }
    }
    else
    {
      // This covariance computation cycle has been skipped, so simply take the
      // last covariance computed
      //
      // We do not propagate the latest covariance forward because it would grow
      // unbounded being very different from the actual covariance we would have
      // computed if not throttling.
      odom_output.pose.covariance = odom_output_.pose.covariance;
      odom_output.twist.covariance = odom_output_.twist.covariance;
      acceleration_output.accel.covariance = acceleration_output_.accel.covariance;
    }
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);

    latest_stamp_ = latest_stamp;
    latest_covariance_stamp_ = latest_covariance_stamp;
    latest_covariance_valid_ = latest_covariance_valid;
    odom_output_ = odom_output;
    acceleration_output_ = acceleration_output;
  }
}

void Odometry3DPublisher::onStart()
{
  synchronizer_ = Synchronizer(device_id_);
  latest_stamp_ = latest_covariance_stamp_ = Synchronizer::TIME_ZERO;
  latest_covariance_valid_ = false;
  odom_output_ = nav_msgs::Odometry();
  acceleration_output_ = geometry_msgs::AccelWithCovarianceStamped();
  publish_timer_.start();
  delayed_throttle_filter_.reset();
}

void Odometry3DPublisher::onStop()
{
  publish_timer_.stop();
}

bool Odometry3DPublisher::getState(const fuse_core::Graph& graph, const ros::Time& stamp,
                                   const fuse_core::UUID& device_id, fuse_core::UUID& position_uuid,
                                   fuse_core::UUID& orientation_uuid, fuse_core::UUID& velocity_linear_uuid,
                                   fuse_core::UUID& velocity_angular_uuid, fuse_core::UUID& acceleration_linear_uuid,
                                   fuse_core::UUID& acceleration_angular_uuid, nav_msgs::Odometry& odometry,
                                   geometry_msgs::AccelWithCovarianceStamped& acceleration)
{
  try
  {
    position_uuid = fuse_variables::Position3DStamped(stamp, device_id).uuid();
    auto position_variable = dynamic_cast<const fuse_variables::Position3DStamped&>(graph.getVariable(position_uuid));

    orientation_uuid = fuse_variables::Orientation3DStamped(stamp, device_id).uuid();
    auto orientation_variable =
        dynamic_cast<const fuse_variables::Orientation3DStamped&>(graph.getVariable(orientation_uuid));

    velocity_linear_uuid = fuse_variables::VelocityLinear3DStamped(stamp, device_id).uuid();
    auto velocity_linear_variable =
        dynamic_cast<const fuse_variables::VelocityLinear3DStamped&>(graph.getVariable(velocity_linear_uuid));

    velocity_angular_uuid = fuse_variables::VelocityAngular3DStamped(stamp, device_id).uuid();
    auto velocity_angular_variable =
        dynamic_cast<const fuse_variables::VelocityAngular3DStamped&>(graph.getVariable(velocity_angular_uuid));

    acceleration_linear_uuid = fuse_variables::AccelerationLinear3DStamped(stamp, device_id).uuid();
    auto acceleration_linear_variable =
        dynamic_cast<const fuse_variables::AccelerationLinear3DStamped&>(graph.getVariable(acceleration_linear_uuid));

    acceleration_angular_uuid = fuse_variables::AccelerationAngular3DStamped(stamp, device_id).uuid();
    auto acceleration_angular_variable =
        dynamic_cast<const fuse_variables::AccelerationAngular3DStamped&>(graph.getVariable(acceleration_angular_uuid));

    odometry.pose.pose.position.x = position_variable.x();
    odometry.pose.pose.position.y = position_variable.y();
    odometry.pose.pose.position.z = position_variable.z();
    odometry.pose.pose.orientation.w = orientation_variable.w();
    odometry.pose.pose.orientation.x = orientation_variable.x();
    odometry.pose.pose.orientation.y = orientation_variable.y();
    odometry.pose.pose.orientation.z = orientation_variable.z();
    odometry.twist.twist.linear.x = velocity_linear_variable.x();
    odometry.twist.twist.linear.y = velocity_linear_variable.y();
    odometry.twist.twist.linear.z = velocity_linear_variable.z();
    odometry.twist.twist.angular.x = velocity_angular_variable.roll();
    odometry.twist.twist.angular.y = velocity_angular_variable.pitch();
    odometry.twist.twist.angular.z = velocity_angular_variable.yaw();

    acceleration.accel.accel.linear.x = acceleration_linear_variable.x();
    acceleration.accel.accel.linear.y = acceleration_linear_variable.y();
    acceleration.accel.accel.linear.z = acceleration_linear_variable.z();
    acceleration.accel.accel.angular.x = acceleration_angular_variable.roll();
    acceleration.accel.accel.angular.y = acceleration_angular_variable.pitch();
    acceleration.accel.accel.angular.z = acceleration_angular_variable.yaw();
  }
  catch (const std::exception& e)
  {
    ROS_WARN_STREAM_THROTTLE(10.0, "Failed to find a state at time " << stamp << ". Error: " << e.what());
    return false;
  }
  catch (...)
  {
    ROS_WARN_STREAM_THROTTLE(10.0, "Failed to find a state at time " << stamp << ". Error: unknown");
    return false;
  }

  return true;
}

void Odometry3DPublisher::publishTimerCallback(const ros::TimerEvent& event)
{
  ros::Time latest_stamp;
  ros::Time latest_covariance_stamp;
  bool latest_covariance_valid;
  nav_msgs::Odometry odom_output;
  geometry_msgs::AccelWithCovarianceStamped acceleration_output;
  {
    std::lock_guard<std::mutex> lock(mutex_);

    latest_stamp = latest_stamp_;
    latest_covariance_stamp = latest_covariance_stamp_;
    latest_covariance_valid = latest_covariance_valid_;
    odom_output = odom_output_;
    acceleration_output = acceleration_output_;
  }

  if (latest_stamp == Synchronizer::TIME_ZERO)
  {
    ROS_WARN_STREAM_FILTER(&delayed_throttle_filter_, "No valid state data yet. Delaying tf broadcast.");
    return;
  }

  tf2::Transform pose;
  tf2::fromMsg(odom_output.pose.pose, pose);

  // If requested, we need to project our state forward in time using the 3D
  // kinematic model
  if (params_.predict_to_current_time)
  {
    tf2::Vector3 velocity_linear;
    tf2::fromMsg(odom_output.twist.twist.linear, velocity_linear);

    tf2::Vector3 velocity_angular;
    tf2::fromMsg(odom_output.twist.twist.angular, velocity_angular);

    const double dt = event.current_real.toSec() - odom_output.header.stamp.toSec();

    fuse_core::Matrix18d jacobian;

    tf2::Vector3 acceleration_linear;
    tf2::Vector3 acceleration_angular;
    if (params_.predict_with_acceleration)
    {
      tf2::fromMsg(acceleration_output.accel.accel.linear, acceleration_linear);
      tf2::fromMsg(acceleration_output.accel.accel.angular, acceleration_angular);
    }

    predict(pose, velocity_linear, velocity_angular, acceleration_linear, acceleration_angular, dt, pose,
            velocity_linear, velocity_angular, acceleration_linear, acceleration_angular, jacobian);

    odom_output.pose.pose.position.x = pose.getOrigin().getX();
    odom_output.pose.pose.position.y = pose.getOrigin().getY();
    odom_output.pose.pose.position.z = pose.getOrigin().getZ();
    odom_output.pose.pose.orientation = tf2::toMsg(pose.getRotation());

    odom_output.twist.twist.linear.x = velocity_linear.x();
    odom_output.twist.twist.linear.y = velocity_linear.y();
    odom_output.twist.twist.linear.z = velocity_linear.z();

    if (params_.predict_with_acceleration)
    {
      acceleration_output.accel.accel.linear.x = acceleration_linear.x();
      acceleration_output.accel.accel.linear.y = acceleration_linear.y();
      acceleration_output.accel.accel.linear.z = acceleration_linear.z();
      acceleration_output.accel.accel.angular.x = acceleration_angular.x();
      acceleration_output.accel.accel.angular.y = acceleration_angular.y();
      acceleration_output.accel.accel.angular.z = acceleration_angular.z();
    }

    odom_output.header.stamp = event.current_real;
    acceleration_output.header.stamp = event.current_real;

    // Either the last covariance computation was skipped because there was no
    // subscriber, or it failed
    if (latest_covariance_valid)
    {
      fuse_core::Matrix18d covariance;
      covariance(0, 0) = odom_output.pose.covariance[0];
      covariance(0, 1) = odom_output.pose.covariance[1];
      covariance(0, 2) = odom_output.pose.covariance[5];
      covariance(1, 0) = odom_output.pose.covariance[6];
      covariance(1, 1) = odom_output.pose.covariance[7];
      covariance(1, 2) = odom_output.pose.covariance[11];
      covariance(2, 0) = odom_output.pose.covariance[30];
      covariance(2, 1) = odom_output.pose.covariance[31];
      covariance(2, 2) = odom_output.pose.covariance[35];

      covariance(3, 3) = odom_output.twist.covariance[0];
      covariance(3, 4) = odom_output.twist.covariance[1];
      covariance(3, 5) = odom_output.twist.covariance[5];
      covariance(4, 3) = odom_output.twist.covariance[6];
      covariance(4, 4) = odom_output.twist.covariance[7];
      covariance(4, 5) = odom_output.twist.covariance[11];
      covariance(5, 3) = odom_output.twist.covariance[30];
      covariance(5, 4) = odom_output.twist.covariance[31];
      covariance(5, 5) = odom_output.twist.covariance[35];

      covariance(6, 6) = acceleration_output.accel.covariance[0];
      covariance(6, 7) = acceleration_output.accel.covariance[1];
      covariance(7, 6) = acceleration_output.accel.covariance[6];
      covariance(7, 7) = acceleration_output.accel.covariance[7];

      // TODO(efernandez) for now we set to zero the out-of-diagonal blocks with
      // the correlations between pose, twist and acceleration, but we could
      // cache them in another attribute when we retrieve the covariance from
      // the ceres problem
      covariance.topRightCorner<3, 5>().setZero();
      covariance.bottomLeftCorner<5, 3>().setZero();
      covariance.block<3, 2>(3, 6).setZero();
      covariance.block<2, 3>(6, 3).setZero();

      covariance = jacobian * covariance * jacobian.transpose();

      geometry_msgs::Twist velocity_linear_twist;
      geometry_msgs::Twist odom_output_twist;

      velocity_linear_twist.linear.x = velocity_linear.getX();
      velocity_linear_twist.linear.y = velocity_linear.getY();
      velocity_linear_twist.linear.z = velocity_linear.getZ();

      auto process_noise_covariance = params_.process_noise_covariance;
      if (params_.scale_process_noise)
      {
        common::scaleProcessNoiseCovariance(process_noise_covariance, velocity_linear_twist, odom_output.twist.twist,
                                            params_.velocity_norm_min);
      }

      covariance.noalias() += dt * process_noise_covariance;

      odom_output.pose.covariance[0] = covariance(0, 0);
      odom_output.pose.covariance[1] = covariance(0, 1);
      odom_output.pose.covariance[5] = covariance(0, 2);
      odom_output.pose.covariance[6] = covariance(1, 0);
      odom_output.pose.covariance[7] = covariance(1, 1);
      odom_output.pose.covariance[11] = covariance(1, 2);
      odom_output.pose.covariance[30] = covariance(2, 0);
      odom_output.pose.covariance[31] = covariance(2, 1);
      odom_output.pose.covariance[35] = covariance(2, 2);

      odom_output.twist.covariance[0] = covariance(3, 3);
      odom_output.twist.covariance[1] = covariance(3, 4);
      odom_output.twist.covariance[5] = covariance(3, 5);
      odom_output.twist.covariance[6] = covariance(4, 3);
      odom_output.twist.covariance[7] = covariance(4, 4);
      odom_output.twist.covariance[11] = covariance(4, 5);
      odom_output.twist.covariance[30] = covariance(5, 3);
      odom_output.twist.covariance[31] = covariance(5, 4);
      odom_output.twist.covariance[35] = covariance(5, 5);

      acceleration_output.accel.covariance[0] = covariance(6, 6);
      acceleration_output.accel.covariance[1] = covariance(6, 7);
      acceleration_output.accel.covariance[6] = covariance(7, 6);
      acceleration_output.accel.covariance[7] = covariance(7, 7);
    }
  }

  odom_pub_.publish(odom_output);
  acceleration_pub_.publish(acceleration_output);

  if (params_.publish_tf)
  {
    auto frame_id = odom_output.header.frame_id;
    auto child_frame_id = odom_output.child_frame_id;

    if (params_.invert_tf)
    {
      pose = pose.inverse();
      std::swap(frame_id, child_frame_id);
    }

    geometry_msgs::TransformStamped trans;
    trans.header.stamp = odom_output.header.stamp;
    trans.header.frame_id = frame_id;
    trans.child_frame_id = child_frame_id;
    trans.transform.translation.x = pose.getOrigin().getX();
    trans.transform.translation.y = pose.getOrigin().getY();
    trans.transform.translation.z = pose.getOrigin().getZ();
    trans.transform.rotation = tf2::toMsg(pose.getRotation());

    if (!params_.invert_tf && params_.world_frame_id == params_.map_frame_id)
    {
      try
      {
        auto base_to_odom = tf_buffer_->lookupTransform(params_.base_link_frame_id, params_.odom_frame_id,
                                                        trans.header.stamp, params_.tf_timeout);

        geometry_msgs::TransformStamped map_to_odom;
        tf2::doTransform(base_to_odom, map_to_odom, trans);
        map_to_odom.child_frame_id = params_.odom_frame_id;
        trans = map_to_odom;
      }
      catch (const std::exception& e)
      {
        ROS_WARN_STREAM_THROTTLE(5.0, "Could not lookup the " << params_.base_link_frame_id << "->"
                                                              << params_.odom_frame_id
                                                              << " transform. Error: " << e.what());

        return;
      }
    }

    tf_broadcaster_.sendTransform(trans);
  }
}

}  // namespace fuse_models
