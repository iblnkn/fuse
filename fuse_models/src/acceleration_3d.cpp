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
#include <fuse_core/transaction.h>
#include <fuse_core/uuid.h>
#include <fuse_models/acceleration_3d.h>
#include <fuse_models/common/sensor_proc.h>
#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

// Register this sensor model as a plugin for the SensorModel class in the fuse_core namespace
PLUGINLIB_EXPORT_CLASS(fuse_models::Acceleration3D, fuse_core::SensorModel)

namespace fuse_models
{
// Define the constructor for the Acceleration3D class
Acceleration3D::Acceleration3D()
  // Initialize the base class and bide the process method to the throttled_callback_
  : fuse_core::AsyncSensorModel(1)
  , device_id_(fuse_core::uuid::NIL)
  , tf_listener_(tf_buffer_)
  , throttled_callback_(std::bind(&Acceleration3D::process, this, std::placeholders::_1))
{
}

// Define the onInit() method, which is called when the plugin is initialized
void Acceleration3D::onInit()
{
  // Load the device ID from the parameter server and the paraters for this plugin from the ROS parameter server.
  device_id_ = fuse_variables::loadDeviceId(private_node_handle_);
  params_.loadFromROS(private_node_handle_);

  // Set the throttle period and use_wall_time of the throttled callback
  throttled_callback_.setThrottlePeriod(params_.throttle_period);
  throttled_callback_.setUseWallTime(params_.throttle_use_wall_time);

  // Check if linear_indices and angular_indices are empty
  if (params_.linear_indices.empty() || params_.angular_indices.empty())
  {
    // Log a warning and ignore the topic if they are
    ROS_WARN_STREAM_NAMED("Acceleration3D", "No dimensions were specified. Data from topic "
                                                << ros::names::resolve(params_.topic) << " will be ignored.");
  }
}

// Define the onStart() method, which is called when the plugin starts receiving data
void Acceleration3D::onStart()
{
  // Check if linear_indices and angular_indices are not empty
  if (!(params_.linear_indices.empty() || params_.angular_indices.empty()))
  {
    // Subscribe to the topic with the given parameters, and use the throttled_callback_ to process the messages
    subscriber_ = node_handle_.subscribe<geometry_msgs::AccelWithCovarianceStamped>(
        ros::names::resolve(params_.topic), params_.queue_size, &AccelerationThrottledCallback::callback,
        &throttled_callback_, ros::TransportHints().tcpNoDelay(params_.tcp_no_delay));
  }
}

// Define the onStop() method, which is called when the plugin stops receiving data
void Acceleration3D::onStop()
{
  // Shutdown the subscriber
  subscriber_.shutdown();
}

// Define the process() method, which is called every time a message is received on the subscribed topic
void Acceleration3D::process(const geometry_msgs::AccelWithCovarianceStamped::ConstPtr& msg)
{
  // Create a transaction object and set the stamp to the header stamp of the received message
  auto transaction = fuse_core::Transaction::make_shared();
  transaction->stamp(msg->header.stamp);

  // Process the received message, and add any resulting variables and constraints to the transaction
  common::processAccel3DWithCovariance(name(), device_id_, *msg, params_.linear_loss, params_.angular_loss,
                                       params_.target_frame, params_.linear_indices, params_.angular_indices,
                                       tf_buffer_, !params_.disable_checks, *transaction, params_.tf_timeout);

  // Send the transaction object to the plugin's parent (i.e., the parent of the AsyncSensorModel class)
  sendTransaction(transaction);
}

}  // namespace fuse_models
