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
#include <fuse_models/skid_steer_3d_predict.h>
#include <fuse_models/skid_steer_3d_state_kinematic_constraint.h>
#include <fuse_models/skid_steer_3d.h>
#include <fuse_models/common/sensor_proc.h>

#include <Eigen/Dense>
#include <fuse_core/async_motion_model.h>
#include <fuse_core/constraint.h>
#include <fuse_core/transaction.h>
#include <fuse_core/uuid.h>
#include <fuse_core/variable.h>
#include <fuse_variables/acceleration_angular_3d_stamped.h>
#include <fuse_variables/acceleration_linear_3d_stamped.h>
#include <fuse_variables/orientation_3d_stamped.h>
#include <fuse_variables/position_3d_stamped.h>
#include <fuse_variables/velocity_angular_3d_stamped.h>
#include <fuse_variables/velocity_linear_3d_stamped.h>
#include <fuse_variables/stamped.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <tf2/utils.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Accel.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Register this motion model with ROS as a plugin.
PLUGINLIB_EXPORT_CLASS(fuse_models::SkidSteer3D, fuse_core::MotionModel)

namespace std
{
inline bool isfinite3d(const geometry_msgs::Pose& pose)
{
  return std::isfinite(pose.position.x) && std::isfinite(pose.position.y) && std::isfinite(pose.position.z) &&
         std::isfinite(pose.orientation.x) && std::isfinite(pose.orientation.y) && std::isfinite(pose.orientation.z) &&
         std::isfinite(pose.orientation.w);
}

inline bool isfinite3d(const geometry_msgs::Twist& twist)
{
  return std::isfinite(twist.linear.x) && std::isfinite(twist.linear.y) && std::isfinite(twist.linear.z) &&
         std::isfinite(twist.angular.x) && std::isfinite(twist.angular.y) && std::isfinite(twist.angular.z);
}

inline bool isfinite3d(const geometry_msgs::Accel& accel)
{
  return std::isfinite(accel.linear.x) && std::isfinite(accel.linear.y) && std::isfinite(accel.linear.z) &&
         std::isfinite(accel.angular.x) && std::isfinite(accel.angular.y) && std::isfinite(accel.angular.z);
}

std::string to_string3d(const geometry_msgs::Pose& pose)
{
  std::ostringstream oss;
  oss << pose;
  return oss.str();
}

std::string to_string3d(const geometry_msgs::Twist& twist)
{
  std::ostringstream oss;
  oss << twist;
  return oss.str();
}

std::string to_string3d(const geometry_msgs::Accel& accel)
{
  std::ostringstream oss;
  oss << accel;
  return oss.str();
}

}  // namespace std

namespace fuse_core
{
template <typename Derived>
inline void validateCovariance(const Eigen::DenseBase<Derived>& covariance,
                               const double precision = Eigen::NumTraits<double>::dummy_precision())
{
  if (!fuse_core::isSymmetric(covariance, precision))
  {
    throw std::runtime_error("Non-symmetric partial covariance matrix\n" +
                             fuse_core::to_string(covariance, Eigen::FullPrecision));
  }

  if (!fuse_core::isPositiveDefinite(covariance))
  {
    throw std::runtime_error("Non-positive-definite partial covariance matrix\n" +
                             fuse_core::to_string(covariance, Eigen::FullPrecision));
  }
}

}  // namespace fuse_core

namespace fuse_models
{
SkidSteer3D::SkidSteer3D()
  : fuse_core::AsyncMotionModel(1)
  , buffer_length_(ros::DURATION_MAX)
  , device_id_(fuse_core::uuid::NIL)
  , timestamp_manager_(&SkidSteer3D::generateMotionModel, this, ros::DURATION_MAX)
{
}

void SkidSteer3D::print(std::ostream& stream) const
{
  stream << "state history:\n";
  for (const auto& state : state_history_)
  {
    stream << "- stamp: " << state.first << "\n";
    state.second.print(stream);
  }
}

void SkidSteer3D::StateHistoryElement::print(std::ostream& stream) const
{
  stream << "  position uuid: " << position_uuid << "\n"
         << "  orientation uuid: " << orientation_uuid << "\n"
         << "  velocity linear uuid: " << vel_linear_uuid << "\n"
         << "  velocity angular uuid: " << vel_angular_uuid << "\n"
         << "  acceleration linear uuid: " << acc_linear_uuid << "\n"
         << "  acceleration angular uuid: " << acc_angular_uuid << "\n"
         << "  pose: " << pose << "\n"
         << "  velocity linear: " << velocity_linear << "\n"
         << "  velocity angular: " << velocity_angular << "\n"
         << "  acceleration linear: " << acceleration_linear << "\n"
         << "  acceleration angular: " << acceleration_angular << "\n";
  ;
}

void SkidSteer3D::StateHistoryElement::validate() const
{
  if (!std::isfinite3d(pose))
  {
    throw std::runtime_error("Invalid pose " + std::to_string3d(pose));
  }

  if (!std::isfinite3d(velocity_linear))
  {
    throw std::runtime_error("Invalid linear velocity " + std::to_string3d(velocity_linear));
  }

  if (!std::isfinite3d(velocity_angular))
  {
    throw std::runtime_error("Invalid angular velocity " + std::to_string3d(velocity_angular));
  }

  if (!std::isfinite3d(acceleration_linear))
  {
    throw std::runtime_error("Invalid linear acceleration " + std::to_string3d(acceleration_linear));
  }
  if (!std::isfinite3d(acceleration_angular))
  {
    throw std::runtime_error("Invalid angular acceleration " + std::to_string3d(acceleration_angular));
  }
}

bool SkidSteer3D::applyCallback(fuse_core::Transaction& transaction)
{
  // Use the timestamp manager to generate just the required motion model segments. The timestamp manager, in turn,
  // makes calls to the generateMotionModel() function.
  try
  {
    // Now actually generate the motion model segments
    timestamp_manager_.query(transaction, true);
  }
  catch (const std::exception& e)
  {
    ROS_ERROR_STREAM_THROTTLE(10.0, "An error occurred while completing the motion model query. Error: " << e.what());
    return false;
  }
  return true;
}

void SkidSteer3D::onGraphUpdate(fuse_core::Graph::ConstSharedPtr graph)
{
  updateStateHistoryEstimates(*graph, state_history_, buffer_length_);
}

void SkidSteer3D::onInit()
{
  std::vector<double> process_noise_diagonal;
  private_node_handle_.param("process_noise_diagonal", process_noise_diagonal, process_noise_diagonal);

  if (process_noise_diagonal.size() != 18)
  {
    throw std::runtime_error("Process noise diagonal must be of length 18!");
  }

  process_noise_covariance_ = fuse_core::Vector18d(process_noise_diagonal.data()).asDiagonal();

  private_node_handle_.param("scale_process_noise", scale_process_noise_, scale_process_noise_);
  private_node_handle_.param("velocity_norm_min", velocity_norm_min_, velocity_norm_min_);

  private_node_handle_.param("disable_checks", disable_checks_, disable_checks_);

  double buffer_length = 3.0;
  private_node_handle_.param("buffer_length", buffer_length, buffer_length);

  if (buffer_length < 0.0)
  {
    throw std::runtime_error("Invalid negative buffer length of " + std::to_string(buffer_length) + " specified.");
  }

  buffer_length_ = (buffer_length == 0.0) ? ros::DURATION_MAX : ros::Duration(buffer_length);
  timestamp_manager_.bufferLength(buffer_length_);

  device_id_ = fuse_variables::loadDeviceId(private_node_handle_);
}

void SkidSteer3D::onStart()
{
  timestamp_manager_.clear();
  state_history_.clear();
}

void SkidSteer3D::generateMotionModel(const ros::Time& beginning_stamp, const ros::Time& ending_stamp,
                                      std::vector<fuse_core::Constraint::SharedPtr>& constraints,
                                      std::vector<fuse_core::Variable::SharedPtr>& variables)
{
  assert(beginning_stamp < ending_stamp || (beginning_stamp == ending_stamp && state_history_.empty()));

  StateHistoryElement base_state;
  ros::Time base_time;

  // Find an entry that is > beginning_stamp
  // The entry that is <= will be the one before it
  auto base_state_pair_it = state_history_.upper_bound(beginning_stamp);
  if (base_state_pair_it == state_history_.begin())
  {
    ROS_WARN_STREAM_COND_NAMED(!state_history_.empty(), "SkidSteerModel",
                               "Unable to locate a state in this history "
                               "with stamp <= "
                                   << beginning_stamp << ". Variables will all be initialized to 0.");
    base_time = beginning_stamp;
  }
  else
  {
    --base_state_pair_it;
    base_time = base_state_pair_it->first;
    base_state = base_state_pair_it->second;
  }

  StateHistoryElement state1;

  // If the nearest state we had was before the beginning stamp, we need to project that state to the beginning stamp
  if (base_time != beginning_stamp)
  {
    predict(base_state.pose, base_state.velocity_linear, base_state.velocity_angular, base_state.acceleration_linear,
            base_state.acceleration_angular, (beginning_stamp - base_time).toSec(), state1.pose, state1.velocity_linear,
            state1.velocity_angular, state1.acceleration_linear, state1.acceleration_angular);
  }
  else
  {
    state1 = base_state;
  }

  // If dt is zero, we only need to update the state history:
  const double dt = (ending_stamp - beginning_stamp).toSec();

  if (dt == 0.0)
  {
    state1.position_uuid = fuse_variables::Position3DStamped(beginning_stamp, device_id_).uuid();
    state1.orientation_uuid = fuse_variables::Orientation3DStamped(beginning_stamp, device_id_).uuid();
    state1.vel_linear_uuid = fuse_variables::VelocityLinear3DStamped(beginning_stamp, device_id_).uuid();
    state1.vel_angular_uuid = fuse_variables::VelocityAngular3DStamped(beginning_stamp, device_id_).uuid();
    state1.acc_linear_uuid = fuse_variables::AccelerationLinear3DStamped(beginning_stamp, device_id_).uuid();
    state1.acc_angular_uuid = fuse_variables::AccelerationAngular3DStamped(beginning_stamp, device_id_).uuid();

    state_history_.emplace(beginning_stamp, std::move(state1));

    return;
  }

  // Now predict to get an initial guess for the state at the ending stamp
  StateHistoryElement state2;
  predict(state1.pose, state1.velocity_linear, state1.velocity_angular, state1.acceleration_linear,
          state1.acceleration_angular, dt, state2.pose, state2.velocity_linear, state2.velocity_angular,
          state2.acceleration_linear, state2.acceleration_angular);

  // Define the fuse variables required for this constraint
  auto position1 = fuse_variables::Position3DStamped::make_shared(beginning_stamp, device_id_);
  auto orientation1 = fuse_variables::Orientation3DStamped::make_shared(beginning_stamp, device_id_);
  auto velocity_linear1 = fuse_variables::VelocityLinear3DStamped::make_shared(beginning_stamp, device_id_);
  auto velocity_angular1 = fuse_variables::VelocityAngular3DStamped::make_shared(beginning_stamp, device_id_);
  auto acceleration_linear1 = fuse_variables::AccelerationLinear3DStamped::make_shared(beginning_stamp, device_id_);
  auto acceleration_angular1 = fuse_variables::AccelerationAngular3DStamped::make_shared(beginning_stamp, device_id_);
  auto position2 = fuse_variables::Position3DStamped::make_shared(ending_stamp, device_id_);
  auto orientation2 = fuse_variables::Orientation3DStamped::make_shared(ending_stamp, device_id_);
  auto velocity_linear2 = fuse_variables::VelocityLinear3DStamped::make_shared(ending_stamp, device_id_);
  auto velocity_angular2 = fuse_variables::VelocityAngular3DStamped::make_shared(ending_stamp, device_id_);
  auto acceleration_linear2 = fuse_variables::AccelerationLinear3DStamped::make_shared(ending_stamp, device_id_);
  auto acceleration_angular2 = fuse_variables::AccelerationAngular3DStamped::make_shared(ending_stamp, device_id_);

  position1->data()[fuse_variables::Position3DStamped::X] = state1.pose.position.x;
  position1->data()[fuse_variables::Position3DStamped::Y] = state1.pose.position.y;
  position1->data()[fuse_variables::Position3DStamped::Z] = state1.pose.position.z;
  orientation1->data()[fuse_variables::Orientation3DStamped::X] = state1.pose.orientation.x;
  orientation1->data()[fuse_variables::Orientation3DStamped::Y] = state1.pose.orientation.y;
  orientation1->data()[fuse_variables::Orientation3DStamped::Z] = state1.pose.orientation.z;
  orientation1->data()[fuse_variables::Orientation3DStamped::W] = state1.pose.orientation.w;
  velocity_linear1->data()[fuse_variables::VelocityLinear3DStamped::X] = state1.velocity_linear.linear.x;
  velocity_linear1->data()[fuse_variables::VelocityLinear3DStamped::Y] = state1.velocity_linear.linear.y;
  velocity_linear1->data()[fuse_variables::VelocityLinear3DStamped::Y] = state1.velocity_linear.linear.z;
  velocity_angular1->data()[fuse_variables::VelocityAngular3DStamped::ROLL] = state1.velocity_angular.angular.x;
  velocity_angular1->data()[fuse_variables::VelocityAngular3DStamped::PITCH] = state1.velocity_angular.angular.x;
  velocity_angular1->data()[fuse_variables::VelocityAngular3DStamped::YAW] = state1.velocity_angular.angular.z;
  acceleration_linear1->data()[fuse_variables::AccelerationLinear3DStamped::X] = state1.acceleration_linear.linear.x;
  acceleration_linear1->data()[fuse_variables::AccelerationLinear3DStamped::Y] = state1.acceleration_linear.linear.y;
  acceleration_linear1->data()[fuse_variables::AccelerationLinear3DStamped::Z] = state1.acceleration_linear.linear.z;
  acceleration_angular1->data()[fuse_variables::AccelerationAngular3DStamped::ROLL] =
      state1.acceleration_linear.angular.x;
  acceleration_angular1->data()[fuse_variables::AccelerationAngular3DStamped::PITCH] =
      state1.acceleration_linear.angular.y;
  acceleration_angular1->data()[fuse_variables::AccelerationAngular3DStamped::YAW] =
      state1.acceleration_linear.angular.z;
  position2->data()[fuse_variables::Position3DStamped::X] = state2.pose.position.x;
  position2->data()[fuse_variables::Position3DStamped::Y] = state2.pose.position.y;
  position2->data()[fuse_variables::Position3DStamped::Z] = state2.pose.position.z;
  orientation2->data()[fuse_variables::Orientation3DStamped::X] = state2.pose.orientation.x;
  orientation2->data()[fuse_variables::Orientation3DStamped::Y] = state2.pose.orientation.y;
  orientation2->data()[fuse_variables::Orientation3DStamped::Z] = state2.pose.orientation.z;
  orientation2->data()[fuse_variables::Orientation3DStamped::W] = state2.pose.orientation.w;
  velocity_linear2->data()[fuse_variables::VelocityLinear3DStamped::X] = state2.velocity_linear.linear.x;
  velocity_linear2->data()[fuse_variables::VelocityLinear3DStamped::Y] = state2.velocity_linear.linear.y;
  velocity_linear2->data()[fuse_variables::VelocityLinear3DStamped::Y] = state2.velocity_linear.linear.z;
  velocity_angular2->data()[fuse_variables::VelocityAngular3DStamped::ROLL] = state2.velocity_angular.angular.x;
  velocity_angular2->data()[fuse_variables::VelocityAngular3DStamped::PITCH] = state2.velocity_angular.angular.x;
  velocity_angular2->data()[fuse_variables::VelocityAngular3DStamped::YAW] = state2.velocity_angular.angular.z;
  acceleration_linear2->data()[fuse_variables::AccelerationLinear3DStamped::X] = state2.acceleration_linear.linear.x;
  acceleration_linear2->data()[fuse_variables::AccelerationLinear3DStamped::Y] = state2.acceleration_linear.linear.y;
  acceleration_linear2->data()[fuse_variables::AccelerationLinear3DStamped::Z] = state2.acceleration_linear.linear.z;
  acceleration_angular2->data()[fuse_variables::AccelerationAngular3DStamped::ROLL] =
      state2.acceleration_linear.angular.x;
  acceleration_angular2->data()[fuse_variables::AccelerationAngular3DStamped::PITCH] =
      state2.acceleration_linear.angular.y;
  acceleration_angular2->data()[fuse_variables::AccelerationAngular3DStamped::YAW] =
      state2.acceleration_linear.angular.z;

  state1.position_uuid = position1->uuid();
  state1.orientation_uuid = orientation1->uuid();
  state1.vel_linear_uuid = velocity_linear1->uuid();
  state1.vel_angular_uuid = velocity_angular1->uuid();
  state1.acc_linear_uuid = acceleration_linear1->uuid();
  state1.acc_angular_uuid = acceleration_angular1->uuid();
  state2.position_uuid = position2->uuid();
  state2.orientation_uuid = orientation2->uuid();
  state2.vel_linear_uuid = velocity_linear2->uuid();
  state2.vel_angular_uuid = velocity_angular2->uuid();
  state2.acc_linear_uuid = acceleration_linear2->uuid();
  state2.acc_angular_uuid = acceleration_angular2->uuid();

  state_history_.emplace(beginning_stamp, std::move(state1));
  state_history_.emplace(ending_stamp, std::move(state2));

  // Scale process noise covariance pose by the norm of the current state twist
  auto process_noise_covariance = process_noise_covariance_;
  if (scale_process_noise_)
  {
    common::scaleProcessNoiseCovariance(process_noise_covariance, state1.velocity_angular, state1.velocity_angular,
                                        velocity_norm_min_);
  }

  // Validate
  process_noise_covariance *= dt;

  if (!disable_checks_)
  {
    try
    {
      validateMotionModel(state1, state2, process_noise_covariance);
    }
    catch (const std::runtime_error& ex)
    {
      ROS_ERROR_STREAM_THROTTLE(10.0, "Invalid '" << name_ << "' motion model: " << ex.what());
      return;
    }
  }

  // Create the constraints for this motion model segment
  auto constraint = fuse_models::SkidSteer3DStateKinematicConstraint::make_shared(
      name(), *position1, *orientation1, *velocity_linear1, *velocity_angular1, *acceleration_linear1,
      *acceleration_angular1, *position2, *orientation2, *velocity_linear2, *velocity_angular2, *acceleration_linear2,
      *acceleration_angular2, process_noise_covariance);

  // Update the output variables
  constraints.push_back(constraint);
  variables.push_back(position1);
  variables.push_back(orientation1);
  variables.push_back(velocity_linear1);
  variables.push_back(velocity_angular1);
  variables.push_back(acceleration_linear1);
  variables.push_back(acceleration_angular1);
  variables.push_back(position2);
  variables.push_back(orientation2);
  variables.push_back(velocity_linear2);
  variables.push_back(velocity_angular2);
  variables.push_back(velocity_angular2);
  variables.push_back(acceleration_linear2);
  variables.push_back(acceleration_angular2);
}

void SkidSteer3D::updateStateHistoryEstimates(const fuse_core::Graph& graph, StateHistory& state_history,
                                              const ros::Duration& buffer_length)
{
  if (state_history.empty())
  {
    return;
  }

  // Compute the expiration time carefully, as ROS can't handle negative times
  const auto& ending_stamp = state_history.rbegin()->first;
  auto expiration_time = ending_stamp.toSec() > buffer_length.toSec() ? ending_stamp - buffer_length : ros::Time(0, 0);

  // Remove state history elements before the expiration time.
  // Be careful to ensure that:
  //  - at least one entry remains at all times
  //  - the history covers *at least* until the expiration time. Longer is acceptable.
  auto expiration_iter = state_history.upper_bound(expiration_time);
  if (expiration_iter != state_history.begin())
  {
    // expiration_iter points to the first element > expiration_time.
    // Back up one entry, to a point that is <= expiration_time
    state_history.erase(state_history.begin(), std::prev(expiration_iter));
  }

  // Update the states in the state history with information from the graph
  // If a state is not in the graph yet, predict the state in question from the closest previous state
  for (auto current_iter = state_history.begin(); current_iter != state_history.end(); ++current_iter)
  {
    const auto& current_stamp = current_iter->first;
    auto& current_state = current_iter->second;
    if (graph.variableExists(current_state.position_uuid) && graph.variableExists(current_state.orientation_uuid) &&
        graph.variableExists(current_state.vel_linear_uuid) && graph.variableExists(current_state.vel_angular_uuid) &&
        graph.variableExists(current_state.acc_linear_uuid) && graph.variableExists(current_state.acc_angular_uuid))
    {
      // This pose does exist in the graph. Update it directly.
      const auto& position = graph.getVariable(current_state.position_uuid);
      const auto& orientation = graph.getVariable(current_state.orientation_uuid);
      const auto& vel_linear = graph.getVariable(current_state.vel_linear_uuid);
      const auto& vel_angular = graph.getVariable(current_state.vel_angular_uuid);
      const auto& acc_linear = graph.getVariable(current_state.acc_linear_uuid);
      const auto& acc_angular = graph.getVariable(current_state.acc_angular_uuid);

      current_state.pose.position.x = (position.data()[fuse_variables::Position3DStamped::X]);
      current_state.pose.position.y = (position.data()[fuse_variables::Position3DStamped::Y]);
      current_state.pose.position.z = (position.data()[fuse_variables::Position3DStamped::Z]);
      current_state.pose.orientation.x = (orientation.data()[fuse_variables::Orientation3DStamped::X]);
      current_state.pose.orientation.y = (orientation.data()[fuse_variables::Orientation3DStamped::Y]);
      current_state.pose.orientation.z = (orientation.data()[fuse_variables::Orientation3DStamped::Z]);
      current_state.pose.orientation.w = (orientation.data()[fuse_variables::Orientation3DStamped::W]);
      current_state.velocity_linear.linear.x = (vel_linear.data()[fuse_variables::VelocityLinear3DStamped::X]);
      current_state.velocity_linear.linear.y = (vel_linear.data()[fuse_variables::VelocityLinear3DStamped::Y]);
      current_state.velocity_linear.linear.z = (vel_linear.data()[fuse_variables::VelocityLinear3DStamped::Z]);
      current_state.velocity_angular.angular.x = vel_angular.data()[fuse_variables::VelocityAngular3DStamped::ROLL];
      current_state.velocity_angular.angular.y = vel_angular.data()[fuse_variables::VelocityAngular3DStamped::PITCH];
      current_state.velocity_angular.angular.z = vel_angular.data()[fuse_variables::VelocityAngular3DStamped::YAW];
      current_state.acceleration_linear.linear.x = (acc_linear.data()[fuse_variables::AccelerationLinear3DStamped::X]);
      current_state.acceleration_linear.linear.y = (acc_linear.data()[fuse_variables::AccelerationLinear3DStamped::Y]);
      current_state.acceleration_linear.linear.z = (acc_linear.data()[fuse_variables::AccelerationLinear3DStamped::Z]);
      current_state.acceleration_angular.angular.x =
          (acc_angular.data()[fuse_variables::AccelerationAngular3DStamped::ROLL]);
      current_state.acceleration_angular.angular.y =
          (acc_angular.data()[fuse_variables::AccelerationAngular3DStamped::PITCH]);
      current_state.acceleration_angular.angular.z =
          (acc_angular.data()[fuse_variables::AccelerationAngular3DStamped::YAW]);
    }
    else if (current_iter != state_history.begin())
    {
      auto previous_iter = std::prev(current_iter);
      const auto& previous_stamp = previous_iter->first;
      const auto& previous_state = previous_iter->second;

      // This state is not in the graph yet, so we can't update/correct the value in our state history. However, the
      // state *before* this one may have been corrected (or one of its predecessors may have been), so we can use
      // that corrected value, along with our prediction logic, to provide a more accurate update to this state.
      predict(previous_state.pose, previous_state.velocity_linear, previous_state.velocity_angular,
              previous_state.acceleration_linear, previous_state.acceleration_angular,
              (current_stamp - previous_stamp).toSec(), current_state.pose, current_state.velocity_linear,
              current_state.velocity_angular, current_state.acceleration_linear, current_state.acceleration_angular);
    }
  }
}

void SkidSteer3D::validateMotionModel(const StateHistoryElement& state1, const StateHistoryElement& state2,
                                      const fuse_core::Matrix18d& process_noise_covariance)
{
  try
  {
    state1.validate();
  }
  catch (const std::runtime_error& ex)
  {
    throw std::runtime_error("Invalid state #1: " + std::string(ex.what()));
  }

  try
  {
    state2.validate();
  }
  catch (const std::runtime_error& ex)
  {
    throw std::runtime_error("Invalid state #2: " + std::string(ex.what()));
  }

  try
  {
    fuse_core::validateCovariance(process_noise_covariance);
  }
  catch (const std::runtime_error& ex)
  {
    throw std::runtime_error("Invalid process noise covariance: " + std::string(ex.what()));
  }
}

std::ostream& operator<<(std::ostream& stream, const SkidSteer3D& skid_steer_3d)
{
  skid_steer_3d.print(stream);
  return stream;
}

}  // namespace fuse_models
