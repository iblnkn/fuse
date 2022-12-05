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
#ifndef FUSE_MODELS_COMMON_SENSOR_PROC_H
#define FUSE_MODELS_COMMON_SENSOR_PROC_H

#include <fuse_constraints/absolute_pose_2d_stamped_constraint.h>
#include <fuse_constraints/relative_pose_2d_stamped_constraint.h>
#include <fuse_constraints/absolute_pose_3d_stamped_constraint.h>
#include <fuse_constraints/relative_pose_3d_stamped_constraint.h>
#include <fuse_constraints/absolute_constraint.h>
#include <fuse_core/eigen.h>
#include <fuse_core/loss.h>
#include <fuse_core/transaction.h>
#include <fuse_core/uuid.h>
#include <fuse_variables/acceleration_linear_2d_stamped.h>
#include <fuse_variables/orientation_2d_stamped.h>
#include <fuse_variables/position_2d_stamped.h>
#include <fuse_variables/velocity_linear_2d_stamped.h>
#include <fuse_variables/velocity_angular_2d_stamped.h>
#include <fuse_variables/acceleration_linear_3d_stamped.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <fuse_variables/velocity_linear_3d_stamped.h>
#include <fuse_variables/velocity_angular_3d_stamped.h>
#include <fuse_variables/stamped.h>

#include <geometry_msgs/AccelWithCovarianceStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Accel.h>

#include <boost/range/join.hpp>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>


namespace tf2
{

/** \brief Apply a geometry_msgs TransformStamped to a geometry_msgs TwistWithCovarianceStamped type.
* This function is a specialization of the doTransform template defined in tf2/convert.h.
* \param t_in The twist to transform, as a timestamped TwistWithCovarianceStamped message.
* \param t_out The transformed twist, as a timestamped TwistWithCovarianceStamped message.
* \param transform The timestamped transform to apply, as a TransformStamped message.
*/
template <>
inline
void doTransform(const geometry_msgs::TwistWithCovarianceStamped& t_in, geometry_msgs::TwistWithCovarianceStamped& t_out, const geometry_msgs::TransformStamped& transform)  // NOLINT
{
  tf2::Vector3 vl;
  fromMsg(t_in.twist.twist.linear, vl);
  tf2::Vector3 va;
  fromMsg(t_in.twist.twist.angular, va);

  tf2::Transform t;
  fromMsg(transform.transform, t);
  t_out.twist.twist.linear = tf2::toMsg(t.getBasis() * vl);
  t_out.twist.twist.angular = tf2::toMsg(t.getBasis() * va);
  t_out.header.stamp = transform.header.stamp;
  t_out.header.frame_id = transform.header.frame_id;

  t_out.twist.covariance = transformCovariance(t_in.twist.covariance, t);
}

/** \brief Apply a geometry_msgs TransformStamped to a geometry_msgs AccelWithCovarianceStamped type.
* This function is a specialization of the doTransform template defined in tf2/convert.h.
* \param t_in The acceleration to transform, as a timestamped AccelWithCovarianceStamped message.
* \param t_out The transformed acceleration, as a timestamped AccelWithCovarianceStamped message.
* \param transform The timestamped transform to apply, as a TransformStamped message.
*/
template <>
inline
void doTransform(const geometry_msgs::AccelWithCovarianceStamped& t_in, geometry_msgs::AccelWithCovarianceStamped& t_out, const geometry_msgs::TransformStamped& transform)  // NOLINT
{
  tf2::Vector3 al;
  fromMsg(t_in.accel.accel.linear, al);
  tf2::Vector3 aa;
  fromMsg(t_in.accel.accel.angular, aa);

  tf2::Transform t;
  fromMsg(transform.transform, t);
  t_out.accel.accel.linear = tf2::toMsg(t.getBasis() * al);
  t_out.accel.accel.angular = tf2::toMsg(t.getBasis() * aa);
  t_out.header.stamp = transform.header.stamp;
  t_out.header.frame_id = transform.header.frame_id;

  t_out.accel.covariance = transformCovariance(t_in.accel.covariance, t);
}

}  // namespace tf2


namespace fuse_models
{

namespace common
{

/**
 * @brief Method to merge two vectors of indices adding an offset to the RHS one.
 *
 * @param[in] lhs_indices - LHS vector of indices
 * @param[in] rhs_indices - RHS vector of indices
 * @param[in] rhs_offset - RHS offset to be added to the RHS vector indices (defaults to 0)
 */
inline std::vector<size_t> mergeIndices(
  const std::vector<size_t>& lhs_indices,
  const std::vector<size_t>& rhs_indices,
  const size_t rhs_offset = 0u)
{
  auto merged_indices = boost::copy_range<std::vector<size_t>>(boost::range::join(lhs_indices, rhs_indices));

  const auto rhs_it = merged_indices.begin() + lhs_indices.size();
  std::transform(
    rhs_it,
    merged_indices.end(),
    rhs_it,
    std::bind(std::plus<size_t>(), std::placeholders::_1, rhs_offset));

  return merged_indices;
}

/**
 * @brief Method to create sub-measurements from full measurements and append them to existing partial measurements
 *
 * @param[in] mean_full - The full mean vector from which we will generate the sub-measurement
 * @param[in] covariance_full - The full covariance matrix from which we will generate the sub-measurement
 * @param[in] indices - The indices we want to include in the sub-measurement
 * @param[in,out] mean_partial - The partial measurement mean to which we want to append
 * @param[in,out] covariance_partial - The partial measurement covariance to which we want to append
 */
inline void populatePartialMeasurement(
  const fuse_core::VectorXd& mean_full,
  const fuse_core::MatrixXd& covariance_full,
  const std::vector<size_t>& indices,
  fuse_core::VectorXd& mean_partial,
  fuse_core::MatrixXd& covariance_partial)
{
  for (size_t r = 0; r < indices.size(); ++r)
  {
    mean_partial(r) = mean_full(indices[r]);

    for (size_t c = 0; c < indices.size(); ++c)
    {
      covariance_partial(r, c) = covariance_full(indices[r], indices[c]);
    }
  }
}

/**
 * @brief Method to validate partial measurements, that checks for finite values and covariance properties
 *
 * @param[in] mean_partial - The partial measurement mean we want to validate
 * @param[in] covariance_partial - The partial measurement covariance we want to validate
 * @param[in] precision - The precision to validate the partial measurements covariance is symmetric
 */
inline void validatePartialMeasurement(
  const fuse_core::VectorXd& mean_partial,
  const fuse_core::MatrixXd& covariance_partial,
  const double precision = Eigen::NumTraits<double>::dummy_precision())
{
  if (!mean_partial.allFinite())
  {
    throw std::runtime_error("Invalid partial mean " + fuse_core::to_string(mean_partial));
  }

  if (!fuse_core::isSymmetric(covariance_partial, precision))
  {
    throw std::runtime_error("Non-symmetric partial covariance matrix\n" +
                             fuse_core::to_string(covariance_partial, Eigen::FullPrecision));
  }

  if (!fuse_core::isPositiveDefinite(covariance_partial))
  {
    throw std::runtime_error("Non-positive-definite partial covariance matrix\n" +
                             fuse_core::to_string(covariance_partial, Eigen::FullPrecision));
  }
}

/**
 * @brief Transforms a ROS geometry message from its frame to the frame of the output message
 *
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] input - The message to transform. Source frame and stamp are dictated by its header.
 * @param[in,out] output - The transformed message. Target frame is dictated by its header.
 * @param [in] timeout - Optional. The maximum time to wait for a transform to become available.
 * @return true if the transform succeeded, false otherwise
 */
template <typename T>
bool transformMessage(
  const tf2_ros::Buffer& tf_buffer,
  const T& input,
  T& output,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  try
  {
    auto trans = geometry_msgs::TransformStamped();
    if (tf_timeout.isZero())
    {
      trans = tf_buffer.lookupTransform(output.header.frame_id, input.header.frame_id, input.header.stamp);
    }
    else
    {
      trans = tf_buffer.lookupTransform(output.header.frame_id, input.header.frame_id, input.header.stamp, tf_timeout);
    }
    tf2::doTransform(input, output, trans);
    return true;
  }
  catch (const tf2::TransformException& ex)
  {
    ROS_WARN_STREAM_DELAYED_THROTTLE(5.0, "Could not transform message from " << input.header.frame_id << " to " <<
      output.header.frame_id << ". Error was " << ex.what());
  }

  return false;
}

/**
 * @brief Extracts 2D pose data from a PoseWithCovarianceStamped message and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (2D position and 2D orientation) and a 2D pose constraint to the given
 * \p transaction. The pose data is extracted from the \p pose message. Only 2D data is used. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose - The PoseWithCovarianceStamped message from which we will extract the pose data
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] target_frame - The frame ID into which the pose data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAbsolutePose2DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  geometry_msgs::PoseWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = pose;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, pose, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform pose message with stamp " << pose.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  // Convert the pose into tf2_2d transform
  geometry_msgs::Pose absolute_pose_2d;

  absolute_pose_2d.position.x = transformed_message.pose.pose.position.x;
  absolute_pose_2d.position.y = transformed_message.pose.pose.position.y;
  absolute_pose_2d.position.z = transformed_message.pose.pose.position.z;

  absolute_pose_2d.orientation.x=transformed_message.pose.pose.orientation.x;
  absolute_pose_2d.orientation.y=transformed_message.pose.pose.orientation.y;
  absolute_pose_2d.orientation.z=transformed_message.pose.pose.orientation.z;
  absolute_pose_2d.orientation.w=transformed_message.pose.pose.orientation.w;

  // Create the pose variable
  auto position = fuse_variables::Position2DStamped::make_shared(pose.header.stamp, device_id);
  auto orientation = fuse_variables::Orientation2DStamped::make_shared(pose.header.stamp, device_id);
  position->x() = absolute_pose_2d.position.x;
  position->y() = absolute_pose_2d.position.y;

  tf2::Quaternion abs_pose_quat;
  double roll;
  double pitch;
  double yaw;
  abs_pose_quat.setX(absolute_pose_2d.orientation.x);
  abs_pose_quat.setY(absolute_pose_2d.orientation.y);
  abs_pose_quat.setZ(absolute_pose_2d.orientation.z);
  abs_pose_quat.setW(absolute_pose_2d.orientation.w);

  tf2::Matrix3x3 m;
  m.setRotation(abs_pose_quat);
  m.getRPY(roll, pitch, yaw);

  orientation->yaw() = yaw;

  // Create the pose for the constraint
  fuse_core::Vector3d pose_mean;
  pose_mean << absolute_pose_2d.position.x, absolute_pose_2d.position.y, yaw;

  // Create the covariance for the constraint
  fuse_core::Matrix3d pose_covariance;
  pose_covariance <<
      transformed_message.pose.covariance[0],
      transformed_message.pose.covariance[1],
      transformed_message.pose.covariance[5],
      transformed_message.pose.covariance[6],
      transformed_message.pose.covariance[7],
      transformed_message.pose.covariance[11],
      transformed_message.pose.covariance[30],
      transformed_message.pose.covariance[31],
      transformed_message.pose.covariance[35];

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_covariance_partial(pose_mean_partial.rows(), pose_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position->size());

  populatePartialMeasurement(pose_mean, pose_covariance, indices, pose_mean_partial, pose_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_mean_partial, pose_covariance_partial);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial absolute pose measurement from '" << source
                                                                                         << "' source: " << ex.what());
      return false;
    }
  }

  // Create an absolute pose constraint
  auto constraint = fuse_constraints::AbsolutePose2DStampedConstraint::make_shared(
    source,
    *position,
    *orientation,
    pose_mean_partial,
    pose_covariance_partial,
    position_indices,
    orientation_indices);

  constraint->loss(loss);

  transaction.addVariable(position);
  transaction.addVariable(orientation);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose.header.stamp);

  return true;
}

/**
 * @brief Extracts 3D pose data from a PoseWithCovarianceStamped message and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (3D position and 3D orientation) and a 3D pose constraint to the given
 * \p transaction. The pose data is extracted from the \p pose message. Only 3D data is used. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose - The PoseWithCovarianceStamped message from which we will extract the pose data
 * @param[in] loss - The loss function for the 3D pose constraint generated
 * @param[in] target_frame - The frame ID into which the pose data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAbsolutePose3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  geometry_msgs::PoseWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = pose;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, pose, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform pose message with stamp " << pose.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  // Create the pose variable
  auto position = fuse_variables::Position3DStamped::make_shared(pose.header.stamp, device_id);
  auto orientation = fuse_variables::Orientation3DStamped::make_shared(pose.header.stamp, device_id);
  position->x() = transformed_message.pose.pose.position.x;
  position->y() = transformed_message.pose.pose.position.y;
  position->z() = transformed_message.pose.pose.position.z;

  orientation->x() = transformed_message.pose.pose.orientation.x;
  orientation->y() = transformed_message.pose.pose.orientation.y;
  orientation->z() = transformed_message.pose.pose.orientation.z;
  orientation->w() = transformed_message.pose.pose.orientation.w;

  // Create the pose for the constraint
  fuse_core::Vector7d pose_mean;
  pose_mean << transformed_message.pose.pose.position.x, transformed_message.pose.pose.position.y, transformed_message.pose.pose.position.z,
               transformed_message.pose.pose.orientation.x, transformed_message.pose.pose.orientation.y, transformed_message.pose.pose.orientation.z ,transformed_message.pose.pose.orientation.w;

  // Create the covariance for the constraint
  fuse_core::Matrix6d pose_covariance;
  pose_covariance <<
    transformed_message.pose.covariance[0],  transformed_message.pose.covariance[1],  transformed_message.pose.covariance[2],  transformed_message.pose.covariance[3],  transformed_message.pose.covariance[4],  transformed_message.pose.covariance[5],
    transformed_message.pose.covariance[6],  transformed_message.pose.covariance[7],  transformed_message.pose.covariance[8],  transformed_message.pose.covariance[9],  transformed_message.pose.covariance[10], transformed_message.pose.covariance[11],
    transformed_message.pose.covariance[12], transformed_message.pose.covariance[13], transformed_message.pose.covariance[14], transformed_message.pose.covariance[15], transformed_message.pose.covariance[16], transformed_message.pose.covariance[17],
    transformed_message.pose.covariance[18], transformed_message.pose.covariance[19], transformed_message.pose.covariance[20], transformed_message.pose.covariance[21], transformed_message.pose.covariance[22], transformed_message.pose.covariance[23],
    transformed_message.pose.covariance[24], transformed_message.pose.covariance[25], transformed_message.pose.covariance[26], transformed_message.pose.covariance[27], transformed_message.pose.covariance[28], transformed_message.pose.covariance[29],
    transformed_message.pose.covariance[30], transformed_message.pose.covariance[31], transformed_message.pose.covariance[32], transformed_message.pose.covariance[33], transformed_message.pose.covariance[34], transformed_message.pose.covariance[35];

  // Create an absolute pose constraint
  auto constraint = fuse_constraints::AbsolutePose3DStampedConstraint::make_shared(
    source,
    *position,
    *orientation,
    pose_mean,
    pose_covariance);

  constraint->loss(loss);

  transaction.addVariable(position);
  transaction.addVariable(orientation);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 2D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. Only 2D data is used. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the covariance of each pose message is rotated into the robot's base frame at the time of
 * pose_absolute1. They are then added in the constraint if the pose measurements are independent.
 * Otherwise, if the pose measurements are dependent, the covariance of pose_absolute1 is substracted from the
 * covariance of pose_absolute2. A small minimum relative covariance is added to avoid getting a zero or
 * ill-conditioned covariance. This could happen if both covariance matrices are the same or very similar, e.g. when
 * pose_absolute1 == pose_absolute2, it's possible that the covariance is the same for both poses.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] independent - Whether the pose measurements are indepent or not
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPose2DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const bool independent,
  const fuse_core::Matrix3d& minimum_pose_relative_covariance,
  const fuse_core::Loss::SharedPtr& loss,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  // Convert the poses into tf2_2d transforms
  geometry_msgs::Pose pose1_2d;

  pose1_2d.position.x = pose1.pose.pose.position.x;
  pose1_2d.position.y = pose1.pose.pose.position.y;
  pose1_2d.position.z = pose1.pose.pose.position.z;

  pose1_2d.orientation.x =pose1.pose.pose.orientation.x;
  pose1_2d.orientation.y =pose1.pose.pose.orientation.y;
  pose1_2d.orientation.z =pose1.pose.pose.orientation.z;
  pose1_2d.orientation.w =pose1.pose.pose.orientation.w;

  tf2::Quaternion abs_pose_quat;
  double pose_1_roll;
  double pose_1_pitch;
  double pose_1_yaw;
  abs_pose_quat.setX(pose1_2d.orientation.x);
  abs_pose_quat.setY(pose1_2d.orientation.y);
  abs_pose_quat.setZ(pose1_2d.orientation.z);
  abs_pose_quat.setW(pose1_2d.orientation.w);

  tf2::Matrix3x3 m;
  m.setRotation(abs_pose_quat);
  m.getRPY(pose_1_roll, pose_1_pitch, pose_1_yaw);

  geometry_msgs::Pose pose2_2d;
  pose2_2d.position.x = pose2.pose.pose.position.x;
  pose2_2d.position.y = pose2.pose.pose.position.y;
  pose2_2d.position.z = pose2.pose.pose.position.z;

  pose2_2d.orientation.x =pose2.pose.pose.orientation.x;
  pose2_2d.orientation.y =pose2.pose.pose.orientation.y;
  pose2_2d.orientation.z =pose2.pose.pose.orientation.z;
  pose2_2d.orientation.w =pose2.pose.pose.orientation.w;

  double pose_2_roll;
  double pose_2_pitch;
  double pose_2_yaw;
  abs_pose_quat.setX(pose2_2d.orientation.x);
  abs_pose_quat.setY(pose2_2d.orientation.y);
  abs_pose_quat.setZ(pose2_2d.orientation.z);
  abs_pose_quat.setW(pose2_2d.orientation.w);

  m.setRotation(abs_pose_quat);
  m.getRPY(pose_2_roll, pose_2_pitch, pose_2_yaw);

  // Create the pose variables
  auto position1 = fuse_variables::Position2DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation2DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1_2d.position.x;
  position1->y() = pose1_2d.position.y;
  orientation1->yaw() = pose_1_yaw;

  auto position2 = fuse_variables::Position2DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation2DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2_2d.position.x;
  position2->y() = pose2_2d.position.y;
  orientation2->yaw() = pose_2_yaw;

  // Create the delta for the constraint
  const double sy = ::sin(-pose_2_yaw);
  const double cy = ::cos(-pose_2_yaw);
  double x_diff = pose2_2d.position.x - pose1_2d.position.x;
  double y_diff = pose2_2d.position.y - pose1_2d.position.y;
  fuse_core::Vector3d pose_relative_mean;
  pose_relative_mean <<
    cy * x_diff - sy * y_diff,
    sy * x_diff + cy * y_diff,
    (pose_2_yaw - pose_1_yaw);

  // Create the covariance components for the constraint
  fuse_core::Matrix3d cov1;
  cov1 <<
    pose1.pose.covariance[0],
    pose1.pose.covariance[1],
    pose1.pose.covariance[5],
    pose1.pose.covariance[6],
    pose1.pose.covariance[7],
    pose1.pose.covariance[11],
    pose1.pose.covariance[30],
    pose1.pose.covariance[31],
    pose1.pose.covariance[35];

  fuse_core::Matrix3d cov2;
  cov2 <<
    pose2.pose.covariance[0],
    pose2.pose.covariance[1],
    pose2.pose.covariance[5],
    pose2.pose.covariance[6],
    pose2.pose.covariance[7],
    pose2.pose.covariance[11],
    pose2.pose.covariance[30],
    pose2.pose.covariance[31],
    pose2.pose.covariance[35];

  fuse_core::Matrix3d pose_relative_covariance;
  if (independent)
  {
    // Compute Jacobians so we can rotate the covariance
    fuse_core::Matrix3d j_pose1;
    j_pose1 <<
      -cy,  sy,  sy * x_diff + cy * y_diff,
      -sy, -cy, -cy * x_diff + sy * y_diff,
        0,   0,                         -1;

    fuse_core::Matrix3d j_pose2;
    j_pose2 <<
       cy, -sy,  0,
       sy,  cy,  0,
        0,   0,  1;

    pose_relative_covariance = j_pose1 * cov1 * j_pose1.transpose() + j_pose2 * cov2 * j_pose2.transpose();
  }
  else
  {
    // For dependent pose measurements p1 and p2, we assume they're computed as:
    //
    // p2 = p1 * p12    [1]
    //
    // where p12 is the relative pose between p1 and p2, which is computed here as:
    //
    // p12 = p1^-1 * p2
    //
    // Note that the twist t12 is computed as:
    //
    // t12 = p12 / dt
    //
    // where dt = t2 - t1, for t1 and t2 being the p1 and p2 timestamps, respectively.
    //
    // The covariance propagation of p2 = p1 * p12 is:
    //
    // C2 = J_p1 * C1 * J_p1^T + J_p12 * C12 * J_p12^T
    //
    // where C1, C2, C12 are the covariance matrices of p1, p2 and dp, respectively, and J_p1 and J_p12 are the
    // jacobians of the equation wrt p1 and p12, respectively.
    //
    // Therefore, the covariance C12 of the relative pose p12 is:
    //
    // C12 = J_p12^-1 * (C2 - J_p1 * C1 * J_p1^T) * J_p12^-T    [2]
    //
    //
    //
    // In SE(2) the poses are represented by:
    //
    //     (R | t)
    // p = (-----)
    //     (0 | 1)
    //
    // where R is the rotation matrix for the yaw angle:
    //
    //     (cos(yaw) -sin(yaw))
    // R = (sin(yaw)  cos(yaw))
    //
    // and t is the translation:
    //
    //     (x)
    // t = (y)
    //
    // The pose composition/multiplication in SE(2) is defined as follows:
    //
    //           (R1 | t1)   (R2 | t2)   (R1 * R2 | R1 * t2 + t1)
    // p1 * p2 = (-------) * (-------) = (----------------------)
    //           ( 0 |  1)   ( 0 |  1)   (      0 |            1)
    //
    // which gives the following equations for each component:
    //
    // x = x2 * cos(yaw1) - y2 * sin(yaw1) + x1
    // y = x2 * sin(yaw1) + y2 * cos(yaw1) + y1
    // yaw = yaw1 + yaw2
    //
    // Since the covariance matrices are defined following that same order for the SE(2) components:
    //
    //     (xx   xy   xyaw  )
    // C = (yx   yy   yyaw  )
    //     (yawx yawy yawyaw)
    //
    // the jacobians must be defined following the same order.
    //
    // The jacobian wrt p1 is:
    //
    //        (1 0 | -sin(yaw1) * x2 - cos(yaw1) * y2)
    // J_p1 = (0 1 |  cos(yaw1) * x2 - sin(yaw1) * y2)
    //        (0 0 |                                1)
    //
    // The jacobian wrt p2 is:
    //
    //        (R1 | 0)   (cos(yaw1) -sin(yaw1) 0)
    // J_p2 = (------) = (sin(yaw1)  cos(yaw1) 0)
    //        ( 0 | 1)   (        0          0 1)
    //
    //
    //
    // Therefore, for the the covariance propagation of [1] we would get the following jacobians:
    //
    //        (1 0 | -sin(yaw1) * x12 - cos(yaw1) * y12)
    // J_p1 = (0 1 |  cos(yaw1) * x12 - sin(yaw1) * y12)
    //        (0 0 |                                  1)
    //
    //         (R1 | 0)   (cos(yaw1) -sin(yaw1) 0)
    // J_p12 = (------) = (sin(yaw1)  cos(yaw1) 0)
    //         ( 0 | 1)   (        0          0 1)
    //
    //
    //
    // At this point we could go one step further since p12 = t12 * dt and include the jacobian of this additional
    // equation:
    //
    // J_t12 = dt * Id
    //
    // where Id is a 3x3 identity matrix.
    //
    // However, that would give us the covariance of the twist t12, and here we simply need the one of the relative
    // pose p12.
    //
    //
    //
    // Finally, since we need the inverse of the jacobian J_p12, we can use the inverse directly:
    //
    //            ( cos(yaw1) sin(yaw1) 0)
    // J_p12^-1 = (-sin(yaw1) cos(yaw1) 0)
    //            (         0         0 1)
    //
    //
    //
    // In the implementation below we use:
    //
    // sy = sin(-yaw1)
    // cy = cos(-yaw1)
    //
    // which are defined before.
    //
    // Therefore, the jacobians end up with the following expressions:
    //
    //        (1 0 | sin(-yaw1) * x12 - cos(-yaw1) * y12)
    // J_p1 = (0 1 | cos(-yaw1) * x12 + sin(-yaw1) * y12)
    //        (0 0 |                                   1)
    //
    //            (cos(-yaw1) -sin(-yaw1) 0)
    // J_p12^-1 = (sin(-yaw1)  cos(-yaw1) 0)
    //            (         0           0 1)
    //
    //
    //
    // Note that the covariance propagation expression derived here for dependent pose measurements gives more accurate
    // results than simply changing the sign in the expression for independent pose measurements, which would be:
    //
    // C12 = J_p2 * C2 * J_p2^T - J_p1 * C1 * J_p1^T
    //
    // where J_p1 and J_p2 are the jacobians for p12 = p1^-1 * p2 (we're abusing the notation here):
    //
    //        (-cos(-yaw1),  sin(-yaw1),  sin(-yaw1) * x12 + cos(-yaw1) * y12)
    // J_p1 = (-sin(-yaw1), -cos(-yaw1), -cos(-yaw1) * x12 + sin(-yaw1) * y12)
    //        (          0,           0,                                   -1)
    //
    //        (R1 | 0)   (cos(yaw1) -sin(yaw1) 0)
    // J_p2 = (------) = (sin(yaw1)  cos(yaw1) 0)
    //        ( 0 | 1)   (        0          0 1)
    //
    // which are the j_pose1 and j_pose2 jacobians used above for the covariance propagation expresion for independent
    // pose measurements.
    //
    // This seems to be the approach adviced in https://github.com/cra-ros-pkg/robot_localization/issues/356, but after
    // comparing the resulting relative pose covariance C12 and the twist covariance, we can conclude that the approach
    // proposed here is the only one that allow us to get results that match.
    //
    // The relative pose covariance C12 and the twist covariance T12 can be compared with:
    //
    // T12 = J_t12 * C12 * J_t12^T
    //
    //
    //
    // In some cases the difference between the C1 and C2 covariance matrices is very small and it could yield to an
    // ill-conditioned C12 covariance. For that reason a minimum covariance is added to [2].
    fuse_core::Matrix3d j_pose1;
    j_pose1 << 1, 0, sy * pose_relative_mean(0) - cy * pose_relative_mean(1),
               0, 1, cy * pose_relative_mean(0) + sy * pose_relative_mean(1),
               0, 0, 1;

    fuse_core::Matrix3d j_pose12_inv;
    j_pose12_inv << cy, -sy, 0,
                    sy,  cy, 0,
                     0,   0, 1;

    pose_relative_covariance = j_pose12_inv * (cov2 - j_pose1 * cov1 * j_pose1.transpose()) * j_pose12_inv.transpose() +
                               minimum_pose_relative_covariance;
  }

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_relative_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_relative_covariance_partial(pose_relative_mean_partial.rows(),
                                                       pose_relative_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position1->size());

  populatePartialMeasurement(
    pose_relative_mean,
    pose_relative_covariance,
    indices,
    pose_relative_mean_partial,
    pose_relative_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_relative_mean_partial, pose_relative_covariance_partial, 1e-6);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial differential pose measurement from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose2DStampedConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean_partial,
    pose_relative_covariance_partial,
    position_indices,
    orientation_indices);

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 3D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. Only 3D data is used. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the covariance of each pose message is rotated into the robot's base frame at the time of
 * pose_absolute1. They are then added in the constraint if the pose measurements are independent.
 * Otherwise, if the pose measurements are dependent, the covariance of pose_absolute1 is substracted from the
 * covariance of pose_absolute2. A small minimum relative covariance is added to avoid getting a zero or
 * ill-conditioned covariance. This could happen if both covariance matrices are the same or very similar, e.g. when
 * pose_absolute1 == pose_absolute2, it's possible that the covariance is the same for both poses.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] independent - Whether the pose measurements are indepent or not
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPose3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const bool independent,
  const fuse_core::Matrix6d& minimum_pose_relative_covariance,
  const fuse_core::Loss::SharedPtr& loss,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  // Create the pose variables
  auto position1 = fuse_variables::Position3DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation3DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() =  pose1.pose.pose.position.x;
  position1->y() =  pose1.pose.pose.position.y;
  position1->z() =  pose1.pose.pose.position.z;
  orientation1->x() = pose1.pose.pose.orientation.x;
  orientation1->y() = pose1.pose.pose.orientation.y;
  orientation1->z() = pose1.pose.pose.orientation.z;
  orientation1->w() = pose1.pose.pose.orientation.w;

  auto position2 = fuse_variables::Position3DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation3DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() =  pose2.pose.pose.position.x;
  position2->y() =  pose2.pose.pose.position.y;
  position2->z() =  pose2.pose.pose.position.z;
  orientation2->x() = pose2.pose.pose.orientation.x;
  orientation2->y() = pose2.pose.pose.orientation.y;
  orientation2->z() = pose2.pose.pose.orientation.z;
  orientation2->w() = pose2.pose.pose.orientation.w;

  // Create the delta for the constraint
  tf2::Vector3 position_diff((pose2.pose.pose.position.x - pose1.pose.pose.position.x),(pose2.pose.pose.position.y - pose1.pose.pose.position.y),(pose2.pose.pose.position.z - pose1.pose.pose.position.z));
  tf2::Quaternion rotation_q( pose1.pose.pose.orientation.x, pose1.pose.pose.orientation.y, pose1.pose.pose.orientation.z, pose1.pose.pose.orientation.w);
  tf2::Matrix3x3 rotation_m(rotation_q);
  fuse_core::Matrix6d rotation_m6; 
  rotation_m6 << rotation_m.getRow(0)[0], rotation_m.getRow(0)[1], rotation_m.getRow(0)[2], 0, 0, 0,
                 rotation_m.getRow(1)[0], rotation_m.getRow(1)[1], rotation_m.getRow(1)[2], 0, 0, 0,
                 rotation_m.getRow(2)[0], rotation_m.getRow(2)[1], rotation_m.getRow(2)[2], 0, 0, 0,
                 0, 0, 0, rotation_m.getRow(0)[0], rotation_m.getRow(0)[1], rotation_m.getRow(0)[2],
                 0, 0, 0, rotation_m.getRow(1)[0], rotation_m.getRow(1)[1], rotation_m.getRow(1)[2],
                 0, 0, 0, rotation_m.getRow(2)[0], rotation_m.getRow(2)[1], rotation_m.getRow(2)[2];

  fuse_core::Matrix6d rotation_m6_t;
  rotation_m6_t =rotation_m6.transpose();
  position_diff = rotation_m*position_diff;
  fuse_core::Vector7d pose_relative_mean;
  pose_relative_mean <<
    position_diff[0],
    position_diff[1],
    position_diff[2],
    (pose2.pose.pose.orientation.x - pose1.pose.pose.orientation.x),
    (pose2.pose.pose.orientation.y - pose1.pose.pose.orientation.y),
    (pose2.pose.pose.orientation.z - pose1.pose.pose.orientation.z),
    (pose2.pose.pose.orientation.w - pose1.pose.pose.orientation.w);

  // Create the covariance components for the constraint
  fuse_core::Matrix6d cov1;
  cov1 <<
    pose1.pose.covariance[0],
    pose1.pose.covariance[1],
    pose1.pose.covariance[5],
    pose1.pose.covariance[6],
    pose1.pose.covariance[7],
    pose1.pose.covariance[11],
    pose1.pose.covariance[30],
    pose1.pose.covariance[31],
    pose1.pose.covariance[35];

  fuse_core::Matrix6d cov2;
  cov2 <<
    pose2.pose.covariance[0],
    pose2.pose.covariance[1],
    pose2.pose.covariance[5],
    pose2.pose.covariance[6],
    pose2.pose.covariance[7],
    pose2.pose.covariance[11],
    pose2.pose.covariance[30],
    pose2.pose.covariance[31],
    pose2.pose.covariance[35];

  fuse_core::Matrix6d pose_relative_covariance;
  fuse_core::Matrix6d cov1_rot;
  fuse_core::Matrix6d cov2_rot;
  cov1_rot = rotation_m6*cov1*rotation_m6_t;
  cov2_rot = rotation_m6*cov2*rotation_m6_t;
  if (independent)
  {
    // Compute Jacobians so we can rotate the covariance

    pose_relative_covariance = cov2_rot + cov1_rot;
  }
  else
  {
    pose_relative_covariance = cov2_rot - cov1_rot +  minimum_pose_relative_covariance;
  }
    
    
  

  

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose3DStampedConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean,
    pose_relative_covariance);

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 2D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. Only 2D data is used. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the twist covariance of the last message is used to compute the relative pose covariance using the time
 * difference between the pose_absolute2 and pose_absolute1 time stamps. This assumes the pose measurements are
 * dependent. A small minimum relative covariance is added to avoid getting a zero or ill-conditioned covariance. This
 * could happen if the twist covariance is very small, e.g. when the twist is zero.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] twist - The second (and temporally later) TwistWithCovarianceStamped message
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] twist_covariance_offset - The twist covariance offset that was added to the twist covariance and must be
 *                                       substracted from it before computing the pose relative covariance from it
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPose2DWithTwistCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Matrix3d& minimum_pose_relative_covariance,
  const fuse_core::Matrix3d& twist_covariance_offset,
  const fuse_core::Loss::SharedPtr& loss,
  const std::vector<size_t>& position_indices,
  const std::vector<size_t>& orientation_indices,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  if (position_indices.empty() && orientation_indices.empty())
  {
    return false;
  }

  // Convert the poses into tf2_2d transforms
  geometry_msgs::Pose pose1_2d;
  pose1_2d.position.x = pose1.pose.pose.position.x;
  pose1_2d.position.y = pose1.pose.pose.position.y;
  pose1_2d.position.z = pose1.pose.pose.position.z;

  pose1_2d.orientation.x = pose1.pose.pose.orientation.x;
  pose1_2d.orientation.y = pose1.pose.pose.orientation.y;
  pose1_2d.orientation.z = pose1.pose.pose.orientation.z;
  pose1_2d.orientation.w = pose1.pose.pose.orientation.w;

  tf2::Quaternion abs_pose_quat;
  double pose1_2d_roll;
  double pose1_2d_pitch;
  double pose1_2d_yaw;
  abs_pose_quat.setX(pose1_2d.orientation.x);
  abs_pose_quat.setY(pose1_2d.orientation.y);
  abs_pose_quat.setZ(pose1_2d.orientation.z);
  abs_pose_quat.setW(pose1_2d.orientation.w);

  tf2::Matrix3x3 m;
  m.setRotation(abs_pose_quat);
  m.getRPY(pose1_2d_roll, pose1_2d_pitch, pose1_2d_yaw);

  geometry_msgs::Pose pose2_2d;

  pose2_2d.position.x = pose2.pose.pose.position.x;
  pose2_2d.position.y = pose2.pose.pose.position.y;
  pose2_2d.position.z = pose2.pose.pose.position.z;

  pose2_2d.orientation.x = pose2.pose.pose.orientation.x;
  pose2_2d.orientation.y = pose2.pose.pose.orientation.y;
  pose2_2d.orientation.z = pose2.pose.pose.orientation.z;
  pose2_2d.orientation.w = pose2.pose.pose.orientation.w;

  double pose2_2d_roll;
  double pose2_2d_pitch;
  double pose2_2d_yaw;
  abs_pose_quat.setX(pose2_2d.orientation.x);
  abs_pose_quat.setY(pose2_2d.orientation.y);
  abs_pose_quat.setZ(pose2_2d.orientation.z);
  abs_pose_quat.setW(pose2_2d.orientation.w);

  m.setRotation(abs_pose_quat);
  m.getRPY(pose2_2d_roll, pose2_2d_pitch, pose2_2d_yaw);

  // Create the pose variables
  auto position1 = fuse_variables::Position2DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 =
    fuse_variables::Orientation2DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1_2d.position.x;
  position1->y() = pose1_2d.position.y;
  orientation1->yaw() = pose1_2d_yaw;

  auto position2 = fuse_variables::Position2DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation2DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2_2d.position.x;
  position2->y() = pose2_2d.position.y;
  orientation2->yaw() = pose1_2d_yaw;

  // Create the delta for the constraint
  tf2::Transform pose1_2d_transform;
  tf2::Transform pose2_2d_transform;

  tf2::Vector3 position2_2d(pose2_2d.position.x, pose2_2d.position.y,0);
  tf2::Quaternion orientation_quat;

  orientation_quat.setRPY(0,0,pose1_2d_yaw);

  pose2_2d_transform.setOrigin(tf2::Vector3(pose2_2d.position.x, pose2_2d.position.y,0));
  pose2_2d_transform.setRotation(orientation_quat);

  position2_2d.setX(pose2_2d.position.x);
  position2_2d.setY(pose2_2d.position.y);
  position2_2d.setZ(0);
  orientation_quat.setRPY(0,0,pose2_2d_yaw);

  pose1_2d_transform.setOrigin(tf2::Vector3(pose1_2d.position.x, pose1_2d.position.y,0));
  pose1_2d_transform.setRotation(orientation_quat);

  const auto delta = pose1_2d_transform.inverseTimes(pose2_2d_transform);

  m.setRotation(orientation_quat);
  double roll;
  double pitch;
  double yaw;

  m.getRPY(roll, pitch, yaw);

  fuse_core::Vector3d pose_relative_mean;
  pose_relative_mean << delta.getOrigin().getX(), delta.getOrigin().getY(), yaw;

  // Create the covariance components for the constraint
  fuse_core::Matrix3d cov;
  cov <<
    twist.twist.covariance[0],
    twist.twist.covariance[1],
    twist.twist.covariance[5],
    twist.twist.covariance[6],
    twist.twist.covariance[7],
    twist.twist.covariance[11],
    twist.twist.covariance[30],
    twist.twist.covariance[31],
    twist.twist.covariance[35];

  // For dependent pose measurements p1 and p2, we assume they're computed as:
  //
  // p2 = p1 * p12    [1]
  //
  // where p12 is the relative pose between p1 and p2, which is computed here as:
  //
  // p12 = p1^-1 * p2
  //
  // Note that the twist t12 is computed as:
  //
  // t12 = p12 / dt
  //
  // where dt = t2 - t1, for t1 and t2 being the p1 and p2 timestamps, respectively.
  //
  // Therefore, the relative pose p12 is computed as follows given the twist t12:
  //
  // p12 = t12 * dt
  //
  // The covariance propagation of this equation is:
  //
  // C12 = J_t12 * T12 * J_t12^T    [2]
  //
  // where T12 is the twist covariance and J_t12 is the jacobian of the equation wrt to t12.
  //
  // The jacobian wrt t12 is:
  //
  // J_t12 = dt * Id
  //
  // where Id is a 3x3 Identity matrix.
  //
  // In some cases the twist covariance T12 is very small and it could yield to an ill-conditioned C12 covariance. For
  // that reason a minimum covariance is added to [2].
  //
  // It is also common that for the same reason, the twist covariance T12 already has a minimum covariance offset added
  // to it by the publisher, so we have to remove it before using it.
  const auto dt = (pose2.header.stamp - pose1.header.stamp).toSec();

  if (dt < 1e-6)
  {
    ROS_ERROR_STREAM_THROTTLE(10.0, "Very small time difference " << dt << "s from '" << source << "' source.");
    return false;
  }

  fuse_core::Matrix3d j_twist;
  j_twist.setIdentity();
  j_twist *= dt;

  fuse_core::Matrix3d pose_relative_covariance =
      j_twist * (cov - twist_covariance_offset) * j_twist.transpose() + minimum_pose_relative_covariance;

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd pose_relative_mean_partial(position_indices.size() + orientation_indices.size());
  fuse_core::MatrixXd pose_relative_covariance_partial(pose_relative_mean_partial.rows(),
                                                       pose_relative_mean_partial.rows());

  const auto indices = mergeIndices(position_indices, orientation_indices, position1->size());

  populatePartialMeasurement(
    pose_relative_mean,
    pose_relative_covariance,
    indices,
    pose_relative_mean_partial,
    pose_relative_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(pose_relative_mean_partial, pose_relative_covariance_partial, 1e-6);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial differential pose measurement using the twist covariance from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose2DStampedConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean_partial,
    pose_relative_covariance_partial,
    position_indices,
    orientation_indices);

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts relative 3D pose data from a PoseWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method computes the delta between two poses and creates the required fuse variables and constraints, and then
 * adds them to the given \p transaction. Only 3D data is used. The pose delta is calculated as
 *
 * pose_relative = pose_absolute1^-1 * pose_absolute2
 *
 * Additionally, the twist covariance of the last message is used to compute the relative pose covariance using the time
 * difference between the pose_absolute2 and pose_absolute1 time stamps. This assumes the pose measurements are
 * dependent. A small minimum relative covariance is added to avoid getting a zero or ill-conditioned covariance. This
 * could happen if the twist covariance is very small, e.g. when the twist is zero.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] pose1 - The first (and temporally earlier) PoseWithCovarianceStamped message
 * @param[in] pose2 - The second (and temporally later) PoseWithCovarianceStamped message
 * @param[in] twist - The second (and temporally later) TwistWithCovarianceStamped message
 * @param[in] minimum_pose_relative_covariance - The minimum pose relative covariance that is always added to the
 *                                               resulting pose relative covariance
 * @param[in] twist_covariance_offset - The twist covariance offset that was added to the twist covariance and must be
 *                                       substracted from it before computing the pose relative covariance from it
 * @param[in] loss - The loss function for the 2D pose constraint generated
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processDifferentialPose3DWithTwistCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::PoseWithCovarianceStamped& pose1,
  const geometry_msgs::PoseWithCovarianceStamped& pose2,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Matrix6d& minimum_pose_relative_covariance,
  const fuse_core::Matrix6d& twist_covariance_offset,
  const fuse_core::Loss::SharedPtr& loss,
  const bool validate,
  fuse_core::Transaction& transaction)
{
  auto position1 = fuse_variables::Position3DStamped::make_shared(pose1.header.stamp, device_id);
  auto orientation1 = fuse_variables::Orientation3DStamped::make_shared(pose1.header.stamp, device_id);
  position1->x() = pose1.pose.pose.position.x;
  position1->y() = pose1.pose.pose.position.y;
  position1->z() = pose1.pose.pose.position.z;
  orientation1->x() = pose1.pose.pose.orientation.x;
  orientation1->y() = pose1.pose.pose.orientation.y;
  orientation1->z() = pose1.pose.pose.orientation.z;
  orientation1->w() = pose1.pose.pose.orientation.w;

  auto position2 = fuse_variables::Position3DStamped::make_shared(pose2.header.stamp, device_id);
  auto orientation2 = fuse_variables::Orientation3DStamped::make_shared(pose2.header.stamp, device_id);
  position2->x() = pose2.pose.pose.position.x;
  position2->y() = pose2.pose.pose.position.y;
  position2->z() = pose2.pose.pose.position.z;
  orientation2->x() = pose2.pose.pose.orientation.x;
  orientation2->y() = pose2.pose.pose.orientation.y;
  orientation2->z() = pose2.pose.pose.orientation.z;
  orientation2->w() = pose2.pose.pose.orientation.w;

  // Create the delta for the constraint
  tf2::Transform pose1_2d_transform;
  tf2::Transform pose2_2d_transform;

  pose1_2d_transform.setOrigin(tf2::Vector3(pose1.pose.pose.position.x, pose1.pose.pose.position.y ,pose1.pose.pose.position.z));
  pose1_2d_transform.setRotation(tf2::Quaternion(pose1.pose.pose.orientation.x, pose1.pose.pose.orientation.y ,pose1.pose.pose.orientation.z, pose1.pose.pose.orientation.w));

  pose2_2d_transform.setOrigin(tf2::Vector3(pose2.pose.pose.position.x, pose2.pose.pose.position.y ,pose2.pose.pose.position.z));
  pose2_2d_transform.setRotation(tf2::Quaternion(pose2.pose.pose.orientation.x, pose2.pose.pose.orientation.y ,pose2.pose.pose.orientation.z, pose2.pose.pose.orientation.w));

  const auto delta = pose1_2d_transform.inverseTimes(pose2_2d_transform);

  fuse_core::Vector7d pose_relative_mean;
  pose_relative_mean << delta.getOrigin().getX(), delta.getOrigin().getY(), delta.getOrigin().getZ(), 
                        delta.getRotation().getX(), delta.getRotation().getY(), delta.getRotation().getZ(), delta.getRotation().getW();

  // Create the covariance components for the constraint
  fuse_core::Matrix6d cov;
  cov <<
    twist.twist.covariance[0],  twist.twist.covariance[1],  twist.twist.covariance[2],  twist.twist.covariance[3],  twist.twist.covariance[4],  twist.twist.covariance[5],
    twist.twist.covariance[6],  twist.twist.covariance[7],  twist.twist.covariance[8],  twist.twist.covariance[9],  twist.twist.covariance[10], twist.twist.covariance[11],
    twist.twist.covariance[12], twist.twist.covariance[13], twist.twist.covariance[14], twist.twist.covariance[15], twist.twist.covariance[16], twist.twist.covariance[17],
    twist.twist.covariance[18], twist.twist.covariance[19], twist.twist.covariance[20], twist.twist.covariance[21], twist.twist.covariance[22], twist.twist.covariance[23],
    twist.twist.covariance[24], twist.twist.covariance[25], twist.twist.covariance[26], twist.twist.covariance[27], twist.twist.covariance[28], twist.twist.covariance[29],
    twist.twist.covariance[30], twist.twist.covariance[31], twist.twist.covariance[32], twist.twist.covariance[33], twist.twist.covariance[34], twist.twist.covariance[35];

  const auto dt = (pose2.header.stamp - pose1.header.stamp).toSec();

  if (dt < 1e-6)
  {
    ROS_ERROR_STREAM_THROTTLE(10.0, "Very small time difference " << dt << "s from '" << source << "' source.");
    return false;
  }

  fuse_core::Matrix6d j_twist;
  j_twist.setIdentity();
  j_twist *= dt;

  fuse_core::Matrix6d pose_relative_covariance =
      j_twist * (cov - twist_covariance_offset) * j_twist.transpose() + minimum_pose_relative_covariance;
  // Create a relative pose constraint.
  auto constraint = fuse_constraints::RelativePose3DStampedConstraint::make_shared(
    source,
    *position1,
    *orientation1,
    *position2,
    *orientation2,
    pose_relative_mean,
    pose_relative_covariance);

  constraint->loss(loss);

  transaction.addVariable(position1);
  transaction.addVariable(orientation1);
  transaction.addVariable(position2);
  transaction.addVariable(orientation2);
  transaction.addConstraint(constraint);
  transaction.addInvolvedStamp(pose1.header.stamp);
  transaction.addInvolvedStamp(pose2.header.stamp);

  return true;
}

/**
 * @brief Extracts velocity data from a TwistWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (2D linear velocity and 2D angular velocity) and their respective
 * constraints to the given \p transaction. The velocity data is extracted from the \p twist message. Only 2D data is
 * used. The data will be automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] twist - The TwistWithCovarianceStamped message from which we will extract the twist data
 * @param[in] linear_velocity_loss - The loss function for the 2D linear velocity constraint generated
 * @param[in] angular_velocity_loss - The loss function for the 2D angular velocity constraint generated
 * @param[in] target_frame - The frame ID into which the twist data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processTwist2DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Loss::SharedPtr& linear_velocity_loss,
  const fuse_core::Loss::SharedPtr& angular_velocity_loss,
  const std::string& target_frame,
  const std::vector<size_t>& linear_indices,
  const std::vector<size_t>& angular_indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do
  if (linear_indices.empty() && angular_indices.empty())
  {
    return false;
  }

  geometry_msgs::TwistWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = twist;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, twist, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform twist message with stamp " << twist.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  bool constraints_added = false;

  // Create two absolute constraints
  if (!linear_indices.empty())
  {
    auto velocity_linear =
      fuse_variables::VelocityLinear2DStamped::make_shared(twist.header.stamp, device_id);
    velocity_linear->x() = transformed_message.twist.twist.linear.x;
    velocity_linear->y() = transformed_message.twist.twist.linear.y;

    // Create the mean twist vectors for the constraints
    fuse_core::Vector2d linear_vel_mean;
    linear_vel_mean << transformed_message.twist.twist.linear.x, transformed_message.twist.twist.linear.y;

    // Create the covariances for the constraints
    fuse_core::Matrix2d linear_vel_covariance;
    linear_vel_covariance <<
        transformed_message.twist.covariance[0],
        transformed_message.twist.covariance[1],
        transformed_message.twist.covariance[6],
        transformed_message.twist.covariance[7];

    // Build the sub-vector and sub-matrices based on the requested indices
    fuse_core::VectorXd linear_vel_mean_partial(linear_indices.size());
    fuse_core::MatrixXd linear_vel_covariance_partial(linear_vel_mean_partial.rows(), linear_vel_mean_partial.rows());

    populatePartialMeasurement(
      linear_vel_mean,
      linear_vel_covariance,
      linear_indices,
      linear_vel_mean_partial,
      linear_vel_covariance_partial);

    bool add_constraint = true;

    if (validate)
    {
      try
      {
        validatePartialMeasurement(linear_vel_mean_partial, linear_vel_covariance_partial);
      }
      catch (const std::runtime_error& ex)
      {
        ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial linear velocity measurement from '"
                                            << source << "' source: " << ex.what());
        add_constraint = false;
      }
    }

    if (add_constraint)
    {
      auto linear_vel_constraint = fuse_constraints::AbsoluteVelocityLinear2DStampedConstraint::make_shared(
        source, *velocity_linear, linear_vel_mean_partial, linear_vel_covariance_partial, linear_indices);

      linear_vel_constraint->loss(linear_velocity_loss);

      transaction.addVariable(velocity_linear);
      transaction.addConstraint(linear_vel_constraint);
      constraints_added = true;
    }
  }

  if (!angular_indices.empty())
  {
    // Create the twist variables
    auto velocity_angular =
      fuse_variables::VelocityAngular2DStamped::make_shared(twist.header.stamp, device_id);
    velocity_angular->yaw() = transformed_message.twist.twist.angular.z;

    fuse_core::Vector1d angular_vel_vector;
    angular_vel_vector << transformed_message.twist.twist.angular.z;

    fuse_core::Matrix1d angular_vel_covariance;
    angular_vel_covariance << transformed_message.twist.covariance[35];

    bool add_constraint = true;

    if (validate)
    {
      try
      {
        validatePartialMeasurement(angular_vel_vector, angular_vel_covariance);
      }
      catch (const std::runtime_error& ex)
      {
        ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial angular velocity measurement from '"
                                            << source << "' source: " << ex.what());
        add_constraint = false;
      }
    }

    if (add_constraint)
    {
      auto angular_vel_constraint = fuse_constraints::AbsoluteVelocityAngular2DStampedConstraint::make_shared(
        source, *velocity_angular, angular_vel_vector, angular_vel_covariance, angular_indices);

      angular_vel_constraint->loss(angular_velocity_loss);

      transaction.addVariable(velocity_angular);
      transaction.addConstraint(angular_vel_constraint);
      constraints_added = true;
    }
  }

  if (constraints_added)
  {
    transaction.addInvolvedStamp(twist.header.stamp);
  }

  return constraints_added;
}

/**
 * @brief Extracts velocity data from a TwistWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds two variables (3D linear velocity and 3D angular velocity) and their respective
 * constraints to the given \p transaction. The velocity data is extracted from the \p twist message. Only 3D data is
 * used. The data will be automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] twist - The TwistWithCovarianceStamped message from which we will extract the twist data
 * @param[in] linear_velocity_loss - The loss function for the 3D linear velocity constraint generated
 * @param[in] angular_velocity_loss - The loss function for the 3D angular velocity constraint generated
 * @param[in] target_frame - The frame ID into which the twist data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processTwist3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::TwistWithCovarianceStamped& twist,
  const fuse_core::Loss::SharedPtr& linear_velocity_loss,
  const fuse_core::Loss::SharedPtr& angular_velocity_loss,
  const std::string& target_frame,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  geometry_msgs::TwistWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = twist;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, twist, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform twist message with stamp " << twist.header.stamp << ". Cannot create constraint.");
      return false;
    }
  }

  bool constraints_added = false;

  // Create two absolute constraints
  auto velocity_linear =
    fuse_variables::VelocityLinear3DStamped::make_shared(twist.header.stamp, device_id);
  velocity_linear->x() = transformed_message.twist.twist.linear.x;
  velocity_linear->y() = transformed_message.twist.twist.linear.y;
  velocity_linear->z() = transformed_message.twist.twist.linear.z;

  // Create the mean twist vectors for the constraints
  fuse_core::Vector3d linear_vel_mean;
  linear_vel_mean << transformed_message.twist.twist.linear.x, transformed_message.twist.twist.linear.y, transformed_message.twist.twist.linear.z;

  // Create the covariances for the constraints
  fuse_core::Matrix6d linear_vel_covariance;
  linear_vel_covariance <<
    transformed_message.twist.covariance[0],  transformed_message.twist.covariance[1],  transformed_message.twist.covariance[2],  transformed_message.twist.covariance[3],  transformed_message.twist.covariance[4],  transformed_message.twist.covariance[5],
    transformed_message.twist.covariance[6],  transformed_message.twist.covariance[7],  transformed_message.twist.covariance[8],  transformed_message.twist.covariance[9],  transformed_message.twist.covariance[10], transformed_message.twist.covariance[11],
    transformed_message.twist.covariance[12], transformed_message.twist.covariance[13], transformed_message.twist.covariance[14], transformed_message.twist.covariance[15], transformed_message.twist.covariance[16], transformed_message.twist.covariance[17],
    transformed_message.twist.covariance[18], transformed_message.twist.covariance[19], transformed_message.twist.covariance[20], transformed_message.twist.covariance[21], transformed_message.twist.covariance[22], transformed_message.twist.covariance[23],
    transformed_message.twist.covariance[24], transformed_message.twist.covariance[25], transformed_message.twist.covariance[26], transformed_message.twist.covariance[27], transformed_message.twist.covariance[28], transformed_message.twist.covariance[29],
    transformed_message.twist.covariance[30], transformed_message.twist.covariance[31], transformed_message.twist.covariance[32], transformed_message.twist.covariance[33], transformed_message.twist.covariance[34], transformed_message.twist.covariance[35];

  // Build the sub-vector and sub-matrices based on the requested indices


  auto linear_vel_constraint = fuse_constraints::AbsoluteVelocityLinear3DStampedConstraint::make_shared(
    source, *velocity_linear, linear_vel_mean, linear_vel_covariance);

  linear_vel_constraint->loss(linear_velocity_loss);

  transaction.addVariable(velocity_linear);
  transaction.addConstraint(linear_vel_constraint);
  constraints_added = true;


  // Create the twist variables
  auto velocity_angular =
    fuse_variables::VelocityAngular3DStamped::make_shared(twist.header.stamp, device_id);
  velocity_angular->yaw() = transformed_message.twist.twist.angular.z;

  fuse_core::Vector1d angular_vel_vector;
  angular_vel_vector << transformed_message.twist.twist.angular.z;

  fuse_core::Matrix1d angular_vel_covariance;
  angular_vel_covariance << transformed_message.twist.covariance[35];

  bool add_constraint = true;

  if (validate)
  {
    try
    {
      validatePartialMeasurement(angular_vel_vector, angular_vel_covariance);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial angular velocity measurement from '"
                                          << source << "' source: " << ex.what());
      add_constraint = false;
    }
    

    if (add_constraint)
    {
      auto angular_vel_constraint = fuse_constraints::AbsoluteVelocityAngular3DStampedConstraint::make_shared(
        source, *velocity_angular, angular_vel_vector, angular_vel_covariance);

      angular_vel_constraint->loss(angular_velocity_loss);

      transaction.addVariable(velocity_angular);
      transaction.addConstraint(angular_vel_constraint);
      constraints_added = true;
    }
  }

  if (constraints_added)
  {
    transaction.addInvolvedStamp(twist.header.stamp);
  }

  return constraints_added;
}

/**
 * @brief Extracts linear acceleration data from an AccelWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds a linear acceleration variable and constraint to the given to the given \p transaction.
 * The acceleration data is extracted from the \p acceleration message. Only 2D data is used. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] acceleration - The AccelWithCovarianceStamped message from which we will extract the acceleration data
 * @param[in] loss - The loss function for the 2D linear acceleration constraint generated
 * @param[in] target_frame - The frame ID into which the acceleration data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAccel2DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::AccelWithCovarianceStamped& acceleration,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const std::vector<size_t>& indices,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do
  if (indices.empty())
  {
    return false;
  }

  geometry_msgs::AccelWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = acceleration;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, acceleration, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform acceleration message with stamp " << acceleration.header.stamp
                                                               << ". Cannot create constraint.");
      return false;
    }
  }

  // Create the acceleration variables
  auto acceleration_linear =
    fuse_variables::AccelerationLinear2DStamped::make_shared(acceleration.header.stamp, device_id);
  acceleration_linear->x() = transformed_message.accel.accel.linear.x;
  acceleration_linear->y() = transformed_message.accel.accel.linear.y;

  // Create the full mean vector and covariance for the constraint
  fuse_core::Vector2d accel_mean;
  accel_mean << transformed_message.accel.accel.linear.x, transformed_message.accel.accel.linear.y;

  fuse_core::Matrix2d accel_covariance;
  accel_covariance <<
      transformed_message.accel.covariance[0],
      transformed_message.accel.covariance[1],
      transformed_message.accel.covariance[6],
      transformed_message.accel.covariance[7];

  // Build the sub-vector and sub-matrices based on the requested indices
  fuse_core::VectorXd accel_mean_partial(indices.size());
  fuse_core::MatrixXd accel_covariance_partial(accel_mean_partial.rows(), accel_mean_partial.rows());

  populatePartialMeasurement(accel_mean, accel_covariance, indices, accel_mean_partial, accel_covariance_partial);

  if (validate)
  {
    try
    {
      validatePartialMeasurement(accel_mean_partial, accel_covariance_partial);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid partial linear acceleration measurement from '"
                                          << source << "' source: " << ex.what());
      return false;
    }
  }

  // Create the constraint
  auto linear_accel_constraint = fuse_constraints::AbsoluteAccelerationLinear2DStampedConstraint::make_shared(
    source,
    *acceleration_linear,
    accel_mean_partial,
    accel_covariance_partial,
    indices);

  linear_accel_constraint->loss(loss);

  transaction.addVariable(acceleration_linear);
  transaction.addConstraint(linear_accel_constraint);
  transaction.addInvolvedStamp(acceleration.header.stamp);

  return true;
}

/**
 * @brief Extracts linear acceleration data from an AccelWithCovarianceStamped and adds that data to a fuse Transaction
 *
 * This method effectively adds a linear acceleration variable and constraint to the given to the given \p transaction.
 * The acceleration data is extracted from the \p acceleration message. Only 3D data is used. The data will be
 * automatically transformed into the \p target_frame before it is used.
 *
 * @param[in] source - The name of the sensor or motion model that generated this constraint
 * @param[in] device_id - The UUID of the machine
 * @param[in] acceleration - The AccelWithCovarianceStamped message from which we will extract the acceleration data
 * @param[in] loss - The loss function for the 3D linear acceleration constraint generated
 * @param[in] target_frame - The frame ID into which the acceleration data will be transformed before it is used
 * @param[in] tf_buffer - The transform buffer with which we will lookup the required transform
 * @param[in] validate - Whether to validate the measurements or not. If the validation fails no constraint is added
 * @param[out] transaction - The generated variables and constraints are added to this transaction
 * @return true if any constraints were added, false otherwise
 */
inline bool processAccel3DWithCovariance(
  const std::string& source,
  const fuse_core::UUID& device_id,
  const geometry_msgs::AccelWithCovarianceStamped& acceleration,
  const fuse_core::Loss::SharedPtr& loss,
  const std::string& target_frame,
  const tf2_ros::Buffer& tf_buffer,
  const bool validate,
  fuse_core::Transaction& transaction,
  const ros::Duration& tf_timeout = ros::Duration(0, 0))
{
  // Make sure we actually have work to do

  geometry_msgs::AccelWithCovarianceStamped transformed_message;
  if (target_frame.empty())
  {
    transformed_message = acceleration;
  }
  else
  {
    transformed_message.header.frame_id = target_frame;

    if (!transformMessage(tf_buffer, acceleration, transformed_message, tf_timeout))
    {
      ROS_WARN_STREAM_DELAYED_THROTTLE(
        10.0,
        "Failed to transform acceleration message with stamp " << acceleration.header.stamp
                                                               << ". Cannot create constraint.");
      return false;
    }
  }

  // Create the acceleration variables
  auto acceleration_linear =
    fuse_variables::AccelerationLinear3DStamped::make_shared(acceleration.header.stamp, device_id);
  acceleration_linear->x() = transformed_message.accel.accel.linear.x;
  acceleration_linear->y() = transformed_message.accel.accel.linear.y;
  acceleration_linear->z() = transformed_message.accel.accel.linear.z;

  // Create the acceleration variables
  auto acceleration_angular =
    fuse_variables::AccelerationAngular3DStamped::make_shared(acceleration.header.stamp, device_id);
  acceleration_angular->roll() = transformed_message.accel.accel.angular.x;
  acceleration_angular->pitch() = transformed_message.accel.accel.angular.y;
  acceleration_angular->yaw() = transformed_message.accel.accel.angular.z;


  // Create the full mean vector and covariance for the constraint
  fuse_core::Vector3d accel_linear_mean;
  fuse_core::Vector3d accel_angular_mean;
  accel_linear_mean << transformed_message.accel.accel.linear.x, transformed_message.accel.accel.linear.y, transformed_message.accel.accel.linear.z;
  accel_angular_mean << transformed_message.accel.accel.angular.x, transformed_message.accel.accel.angular.y, transformed_message.accel.accel.angular.z;

  fuse_core::Matrix3d accel_linear_covariance;
  accel_linear_covariance <<
      transformed_message.accel.covariance[0],
      transformed_message.accel.covariance[1],
      transformed_message.accel.covariance[2],
      transformed_message.accel.covariance[6],
      transformed_message.accel.covariance[7],
      transformed_message.accel.covariance[8],
      transformed_message.accel.covariance[12],
      transformed_message.accel.covariance[13],
      transformed_message.accel.covariance[14];

  
  fuse_core::Matrix3d accel_angular_covariance;
  accel_angular_covariance <<
      transformed_message.accel.covariance[21],
      transformed_message.accel.covariance[22],
      transformed_message.accel.covariance[23],
      transformed_message.accel.covariance[27],
      transformed_message.accel.covariance[28],
      transformed_message.accel.covariance[29],
      transformed_message.accel.covariance[33],
      transformed_message.accel.covariance[34],
      transformed_message.accel.covariance[35];

  // Build the sub-vector and sub-matrices based on the requested indices

  // Create the constraint
  auto linear_accel_constraint = fuse_constraints::AbsoluteAccelerationLinear3DStampedConstraint::make_shared(
    source,
    *acceleration_linear,
    accel_linear_mean,
    accel_linear_covariance);

  auto angular_accel_constraint = fuse_constraints::AbsoluteAccelerationAngular3DStampedConstraint::make_shared(
    source,
    *acceleration_angular,
    accel_angular_mean,
    accel_angular_covariance);

  linear_accel_constraint->loss(loss);

  transaction.addVariable(acceleration_linear);
  transaction.addConstraint(linear_accel_constraint);
  transaction.addInvolvedStamp(acceleration.header.stamp);

  return true;
}

/**
 * @brief Scales the process noise covariance pose by the norm of the velocity
 *
 * @param[in, out] process_noise_covariance - The process noise covariance to scale. Only the pose components (x, y,
 *                                            yaw) are scaled, and they are assumed to be in the top left 3x3 corner
 * @param[in] velocity_linear - The linear velocity
 * @param[in] velocity_yaw - The yaw velocity
 * @param[in] velocity_norm_min - The minimum velocity norm
 */
inline void scaleProcessNoiseCovariance(fuse_core::Matrix8d& process_noise_covariance,
                                        const geometry_msgs::Twist& velocity_linear, const double velocity_yaw,
                                        const double velocity_norm_min)
{
  // A more principled approach would be to get the current velocity from the state, make a diagonal matrix from it,
  // and then rotate it to be in the world frame (i.e., the same frame as the pose data). We could then use this
  // rotated velocity matrix to scale the process noise covariance for the pose variables as
  // rotatedVelocityMatrix * poseCovariance * rotatedVelocityMatrix'
  // However, this presents trouble for robots that may incur rotational error as a result of linear motion (and
  // vice-versa). Instead, we create a diagonal matrix whose diagonal values are the vector norm of the state's
  // velocity. We use that to scale the process noise covariance.
  //
  // The comment above has been taken from:
  // https://github.com/cra-ros-pkg/robot_localization/blob/melodic-devel/src/filter_base.cpp#L138-L144
  //
  // We also need to make sure the norm is not zero, because otherwise the resulting process noise covariance for the
  // pose becomes zero and we get NaN when we compute the inverse to obtain the information
  fuse_core::Matrix3d velocity;
  velocity.setIdentity();
  velocity.diagonal() *=
      std::max(velocity_norm_min, fuse_core::Vector3d(velocity_linear.linear.x, velocity_linear.linear.y, velocity_yaw).norm());

  process_noise_covariance.topLeftCorner<3, 3>() =
      velocity * process_noise_covariance.topLeftCorner<3, 3>() * velocity.transpose();
}
inline void scaleProcessNoiseCovariance(fuse_core::Matrix15d& process_noise_covariance,
                                        const geometry_msgs::Twist& velocity_in,
                                        const double velocity_norm_min)
{
  fuse_core::Matrix6d velocity;
  velocity.setIdentity();
  velocity.diagonal() *=
      std::max(velocity_norm_min, pow((pow(velocity_in.linear.x, 2) + pow(velocity_in.linear.y, 2)+ pow(velocity_in.linear.z, 2)+ pow(velocity_in.angular.x, 2)+ pow(velocity_in.angular.y, 2)+ pow(velocity_in.angular.z, 2)),0.5));
      //TODO:: While this technically should work. It is not very pretty. Probably better to take norm of the linear norm and the angular norm.  
  process_noise_covariance.topLeftCorner<6, 6>() =
      velocity * process_noise_covariance.topLeftCorner<6, 6>() * velocity.transpose();
}
inline void scaleProcessNoiseCovariance(fuse_core::Matrix18d& process_noise_covariance,
                                        const geometry_msgs::Twist& velocity_in,
                                        const double velocity_norm_min)
{
  fuse_core::Matrix6d velocity;
  velocity.setIdentity();
  velocity.diagonal() *=
      std::max(velocity_norm_min, pow((pow(velocity_in.linear.x, 2) + pow(velocity_in.linear.y, 2)+ pow(velocity_in.linear.z, 2)+ pow(velocity_in.angular.x, 2)+ pow(velocity_in.angular.y, 2)+ pow(velocity_in.angular.z, 2)),0.5));
//TODO:: While this technically should work. It is not very pretty. Probably better to take norm of the linear norm and the angular norm.  
  process_noise_covariance.topLeftCorner<6, 6>() =
      velocity * process_noise_covariance.topLeftCorner<6, 6>() * velocity.transpose();
}

inline void scaleProcessNoiseCovariance(fuse_core::Matrix18d& process_noise_covariance,
                                        const geometry_msgs::Twist& velocity_linear,
                                        const geometry_msgs::Twist& velocity_angular,
                                        const double velocity_norm_min)
{
  fuse_core::Matrix6d velocity;
  velocity.setIdentity();
  velocity.diagonal() *=
      std::max(velocity_norm_min, pow((pow(velocity_linear.linear.x, 2) + pow(velocity_linear.linear.y, 2)+ pow(velocity_linear.linear.z, 2)+ pow(velocity_angular.angular.x, 2)+ pow(velocity_angular.angular.y, 2)+ pow(velocity_angular.angular.z, 2)),0.5));
//TODO:: While this technically should work. It is not very pretty. Probably better to take norm of the linear norm and the angular norm.  
  process_noise_covariance.topLeftCorner<6, 6>() =
      velocity * process_noise_covariance.topLeftCorner<6, 6>() * velocity.transpose();
}

// /**
//  * @brief Scales the process noise covariance pose by the norm of the velocity
//  *
//  * @param[in, out] process_noise_covariance - The process noise covariance to scale. Only the pose components (x, y,
//  *                                            yaw) are scaled, and they are assumed to be in the top left 3x3 corner
//  * @param[in] velocity_linear - The linear velocity
//  * @param[in] velocity_yaw - The yaw velocity
//  * @param[in] velocity_norm_min - The minimum velocity norm
//  */
// inline void scaleProcessNoiseCovariance(fuse_core::Matrix15d& process_noise_covariance,
//                                         const geometry_msgs::Twist& velocity,
//                                         const double velocity_norm_min)
// {
//   // A more principled approach would be to get the current velocity from the state, make a diagonal matrix from it,
//   // and then rotate it to be in the world frame (i.e., the same frame as the pose data). We could then use this
//   // rotated velocity matrix to scale the process noise covariance for the pose variables as
//   // rotatedVelocityMatrix * poseCovariance * rotatedVelocityMatrix'
//   // However, this presents trouble for robots that may incur rotational error as a result of linear motion (and
//   // vice-versa). Instead, we create a diagonal matrix whose diagonal values are the vector norm of the state's
//   // velocity. We use that to scale the process noise covariance.
//   //
//   // The comment above has been taken from:
//   // https://github.com/cra-ros-pkg/robot_localization/blob/melodic-devel/src/filter_base.cpp#L138-L144
//   //
//   // We also need to make sure the norm is not zero, because otherwise the resulting process noise covariance for the
//   // pose becomes zero and we get NaN when we compute the inverse to obtain the information
//   fuse_core::Matrix6d velocity_m;
//   fuse_core::Vector6d velocity_vect;
//   velocity_vect<<velocity.linear.x, velocity.linear.y, velocity.linear.z, velocity.angular.x, velocity.angular.y, velocity.angular.z;
//   velocity_m.setIdentity();
//   velocity_m.diagonal() *=
//       std::max(velocity_norm_min, velocity_vect.norm());

//   process_noise_covariance.topLeftCorner<3, 3>() =
//       velocity_m * process_noise_covariance.topLeftCorner<3, 3>() * velocity_m.transpose();
// }

}  // namespace common

}  // namespace fuse_models

#endif  // FUSE_MODELS_COMMON_SENSOR_PROC_H
