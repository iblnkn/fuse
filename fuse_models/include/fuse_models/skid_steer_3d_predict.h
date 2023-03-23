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
#ifndef FUSE_MODELS_SKID_STEER_3D_PREDICT_H
#define FUSE_MODELS_SKID_STEER_3D_PREDICT_H

#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <fuse_core/util.h>
#include <fuse_core/eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/Accel.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <array>

namespace fuse_models
{
/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] position1_z - First Y position
 * @param[in] orientation1_w - First W orientation
 * @param[in] orientation1_x - First X orientation
 * @param[in] orientation1_y - First Y orientation
 * @param[in] orientation1_z - First Z orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_linear1_z - First Y velocity
 * @param[in] vel_angular1_x - First yaw velocity
 * @param[in] vel_angular1_y - First yaw velocity
 * @param[in] vel_angular1_z - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] acc_linear1_z - First Y acceleration
 * @param[in] acc_angular1_x - First X acceleration
 * @param[in] acc_angular1_y - First Y acceleration
 * @param[in] acc_angular1_z - First Y acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] position2_z - Second Y position
 * @param[out] orientation2_w - Second W orientation
 * @param[out] orientation2_x - Second X orientation
 * @param[out] orientation2_y - Second Y orientation
 * @param[out] orientation2_z - Second Z orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_linear2_z - Second Y velocity
 * @param[out] vel_angular2_x - Second yaw velocity
 * @param[out] vel_angular2_y - Second yaw velocity
 * @param[out] vel_angular2_z - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 * @param[out] acc_linear2_z - Second Y acceleration
 * @param[out] acc_angular2_x - Second X acceleration
 * @param[out] acc_angular2_y - Second Y acceleration
 * @param[out] acc_angular2_z - Second Y acceleration
 */
template <typename T>
inline void predict(const T position1_x, const T position1_y, const T position1_z, const T orientation1_w,
                    const T orientation1_x, const T orientation1_y, const T orientation1_z, const T vel_linear1_x,
                    const T vel_linear1_y, const T vel_linear1_z, const T vel_angular1_x, const T vel_angular1_y,
                    const T vel_angular1_z, const T acc_linear1_x, const T acc_linear1_y, const T acc_linear1_z,
                    const T acc_angular1_x, const T acc_angular1_y, const T acc_angular1_z, const T dt, T& position2_x,
                    T& position2_y, T& position2_z, T& orientation2_w, T& orientation2_x, T& orientation2_y,
                    T& orientation2_z, T& vel_linear2_x, T& vel_linear2_y, T& vel_linear2_z, T& vel_angular2_x,
                    T& vel_angular2_y, T& vel_angular2_z, T& acc_linear2_x, T& acc_linear2_y, T& acc_linear2_z,
                    T& acc_angular2_x, T& acc_angular2_y, T& acc_angular2_z)
{
  T orientation1_quat[4];
  T orientation1_euler[3];
  orientation1_quat[0] = orientation1_w;
  orientation1_quat[1] = orientation1_x;
  orientation1_quat[2] = orientation1_y;
  orientation1_quat[3] = orientation1_z;
  ceres::QuaternionToAngleAxis(orientation1_quat, orientation1_euler);

  const T orientation1_roll = orientation1_euler[0];
  const T orientation1_pitch = orientation1_euler[1];
  const T orientation1_yaw = orientation1_euler[2];

  // There are better models for this projection, but this matches the one used by r_l.
  T delta_x = vel_linear1_x * dt + T(0.5) * acc_linear1_x * dt * dt;
  T delta_y = vel_linear1_y * dt + T(0.5) * acc_linear1_y * dt * dt;
  T delta_z = vel_linear1_z * dt + T(0.5) * acc_linear1_z * dt * dt;

  T delta_roll = vel_angular1_x * dt + T(0.5) * acc_angular1_x * dt * dt;
  T delta_pitch = vel_angular1_y * dt + T(0.5) * acc_angular1_y * dt * dt;
  T delta_yaw = vel_angular1_z * dt + T(0.5) * acc_angular1_z * dt * dt;

  T sr = ceres::sin(orientation1_roll);
  T cr = ceres::cos(orientation1_roll);

  T sp = ceres::sin(orientation1_pitch);
  T cp = ceres::cos(orientation1_pitch);

  T cpi = T(1) / cp;
  T tp = sp * cpi;

  T sy = ceres::sin(orientation1_yaw);
  T cy = ceres::cos(orientation1_yaw);

  position2_x =
      position1_x + (cy * cp) * delta_x + (cy * sp * sr - sy * cr) * delta_y + (cy * sp * cr + sy * sr) * delta_z;
  position2_y =
      position1_y + (sy * cp) * delta_x + (sy * sp * sr + cy * cr) * delta_y + (sy * sp * cr - cy * sr) * delta_z;
  position2_z = position1_z + (-sp) * delta_x + (cp * sr) * delta_y + (cp * cr) * delta_z;

  T orientation2_roll = orientation1_roll + delta_roll + (sr * tp) * delta_pitch + (cr * tp) * delta_yaw;
  T orientation2_pitch = orientation1_pitch + (cr)*delta_pitch + (-sr) * delta_yaw;
  T orientation2_yaw = orientation1_yaw + (sr * cpi) * delta_pitch + (cr * cpi) * delta_yaw;

  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_linear2_z = vel_linear1_z + acc_linear1_z * dt;

  vel_angular2_x = vel_angular1_x + acc_angular1_x * dt;
  vel_angular2_y = vel_angular1_y + acc_angular1_y * dt;
  vel_angular2_z = vel_angular1_z + acc_angular1_z * dt;

  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;
  acc_linear2_z = acc_linear1_z;

  acc_angular2_x = acc_angular1_x;
  acc_angular2_y = acc_angular1_y;
  acc_angular2_z = acc_angular1_z;

  T orientation2_euler[3];
  T orientation2_quat[4];

  fuse_core::wrapAngle(orientation2_roll);
  fuse_core::wrapAngle(orientation2_pitch);
  fuse_core::wrapAngle(orientation2_yaw);

  orientation2_euler[0] = orientation2_roll;
  orientation2_euler[1] = orientation2_pitch;
  orientation2_euler[2] = orientation2_yaw;

  ceres::AngleAxisToQuaternion(orientation2_euler, orientation2_quat);
  orientation2_w = orientation2_quat[0];
  orientation2_x = orientation2_quat[1];
  orientation2_y = orientation2_quat[2];
  orientation2_z = orientation2_quat[3];

  fuse_core::wrapAngle(vel_angular1_x);
  fuse_core::wrapAngle(vel_angular1_y);
  fuse_core::wrapAngle(vel_angular1_z);

  fuse_core::wrapAngle(acc_angular1_x);
  fuse_core::wrapAngle(acc_angular1_y);
  fuse_core::wrapAngle(acc_angular1_z);
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] position1_z - First Y position
 * @param[in] orientation1_w - First orientation
 * @param[in] orientation1_x - First orientation
 * @param[in] orientation1_y - First orientation
 * @param[in] orientation1_z - First orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_linear1_z - First Y velocity
 * @param[in] vel_angular1_x - First yaw velocity
 * @param[in] vel_angular1_y - First yaw velocity
 * @param[in] vel_angular1_z - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] acc_linear1_z - First Y acceleration
 * @param[in] acc_angular1_x - First X acceleration
 * @param[in] acc_angular1_y - First Y acceleration
 * @param[in] acc_angular1_z - First Y acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] position2_z - Second Y position
 * @param[out] orientation2_w - Second orientation
 * @param[out] orientation2_x - Second orientation
 * @param[out] orientation2_y - Second orientation
 * @param[out] orientation2_z - Second orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_linear2_z - Second Y velocity
 * @param[out] vel_angular2_x - Second yaw velocity
 * @param[out] vel_angular2_y - Second yaw velocity
 * @param[out] vel_angular2_z - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 * @param[out] acc_linear2_z - Second Y acceleration
 * @param[out] acc_angular2_x - Second X acceleration
 * @param[out] acc_angular2_y - Second Y acceleration
 * @param[out] acc_angular2_z - Second Y acceleration
 * @param[out] jacobians - Jacobians wrt the state
 */
template <typename T>
inline void predict(const T position1_x, const T position1_y, const T position1_z, const T orientation1_w,
                    const T orientation1_x, const T orientation1_y, const T orientation1_z, const T vel_linear1_x,
                    const T vel_linear1_y, const T vel_linear1_z, const T vel_angular1_x, const T vel_angular1_y,
                    const T vel_angular1_z, const T acc_linear1_x, const T acc_linear1_y, const T acc_linear1_z,
                    const T acc_angular1_x, const T acc_angular1_y, const T acc_angular1_z, const T dt, T& position2_x,
                    T& position2_y, T& position2_z, T& orientation2_w, T& orientation2_x, T& orientation2_y,
                    T& orientation2_z, T& vel_linear2_x, T& vel_linear2_y, T& vel_linear2_z, T& vel_angular2_x,
                    T& vel_angular2_y, T& vel_angular2_z, T& acc_linear2_x, T& acc_linear2_y, T& acc_linear2_z,
                    T& acc_angular2_x, T& acc_angular2_y, T& acc_angular2_z, T** jacobians)

{
  T orientation1_quat[4];
  T orientation1_euler[3];
  orientation1_quat[0] = orientation1_w;
  orientation1_quat[1] = orientation1_x;
  orientation1_quat[2] = orientation1_y;
  orientation1_quat[3] = orientation1_z;
  ceres::QuaternionToAngleAxis(orientation1_quat, orientation1_euler);

  const T orientation1_roll = orientation1_euler[0];
  const T orientation1_pitch = orientation1_euler[1];
  const T orientation1_yaw = orientation1_euler[2];

  T delta_x = vel_linear1_x * dt + T(0.5) * acc_linear1_x * dt * dt;
  T delta_y = vel_linear1_y * dt + T(0.5) * acc_linear1_y * dt * dt;
  T delta_z = vel_linear1_z * dt + T(0.5) * acc_linear1_z * dt * dt;

  T delta_roll = vel_angular1_x * dt + T(0.5) * acc_angular1_x * dt * dt;
  T delta_pitch = vel_angular1_y * dt + T(0.5) * acc_angular1_y * dt * dt;
  T delta_yaw = vel_angular1_z * dt + T(0.5) * acc_angular1_z * dt * dt;

  T sr = ceres::sin(orientation1_roll);
  T cr = ceres::cos(orientation1_roll);

  T sp = ceres::sin(orientation1_pitch);
  T cp = ceres::cos(orientation1_pitch);

  T cpi = 1 / cp;
  T tp = sp * cpi;

  T sy = ceres::sin(orientation1_yaw);
  T cy = ceres::cos(orientation1_yaw);

  position2_x =
      position1_x + (cy * cp) * delta_x + (cy * sp * sr - sy * cr) * delta_y + (cy * sp * cr + sy * sr) * delta_z;
  position2_y =
      position1_y + (sy * cp) * delta_x + (sy * sp * sr + cy * cr) * delta_y + (sy * sp * cr - cy * sr) * delta_z;
  position2_z = position1_z + (-sp) * delta_x + (cp * sr) * delta_y + (cp * cr) * delta_z;

  T orientation2_roll = orientation1_roll + delta_roll + (sr * tp) * delta_pitch + (cr * tp) * delta_yaw;
  T orientation2_pitch = orientation1_pitch + (cr)*delta_pitch + (-sr) * delta_yaw;
  T orientation2_yaw = orientation1_yaw + (sr * cpi) * delta_pitch + (cr * cpi) * delta_yaw;

  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_linear2_z = vel_linear1_z + acc_linear1_z * dt;

  vel_angular2_x = vel_angular1_x + acc_angular1_x * dt;
  vel_angular2_y = vel_angular1_y + acc_angular1_y * dt;
  vel_angular2_z = vel_angular1_z + acc_angular1_z * dt;

  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;
  acc_linear2_z = acc_linear1_z;

  acc_angular2_x = acc_angular1_x;
  acc_angular2_y = acc_angular1_y;
  acc_angular2_z = acc_angular1_z;

  T orientation2_euler[3];
  T orientation2_quat[4];

  fuse_core::wrapAngle(orientation2_roll);
  fuse_core::wrapAngle(orientation2_pitch);
  fuse_core::wrapAngle(orientation2_yaw);

  orientation2_euler[0] = orientation2_roll;
  orientation2_euler[1] = orientation2_pitch;
  orientation2_euler[2] = orientation2_yaw;

  ceres::AngleAxisToQuaternion(orientation2_euler, orientation2_quat);
  orientation2_w = orientation2_quat[0];
  orientation2_x = orientation2_quat[1];
  orientation2_y = orientation2_quat[2];
  orientation2_z = orientation2_quat[3];

  fuse_core::wrapAngle(vel_angular1_x);
  fuse_core::wrapAngle(vel_angular1_y);
  fuse_core::wrapAngle(vel_angular1_z);

  fuse_core::wrapAngle(acc_angular1_x);
  fuse_core::wrapAngle(acc_angular1_y);
  fuse_core::wrapAngle(acc_angular1_z);

  if (jacobians)
  {
    // Jacobian wrt position1
    if (jacobians[0])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[0]);
      jacobian << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }

    if (jacobians[1])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[1]);
      jacobian << (sy * sr + cy * sp * cr) * delta_y + (sy * sr - cy * sp * sr) * delta_z,
          (-cy * sp * delta_x)(cy * cp * sr) * delta_y + (cy * cp * cr) * delta_z,
          -sy * cp * delta_x + (-cy * cr - sy * sp * sr) * delta_y + (cy * sr - sy * sp * cr) * delta_z,
          (-cy * sr + sy * sp * cr) * delta_y + (-cy * cr - sy * sp * sr) * delta_z,
          -sy * sp * delta_x + sy * cp * sr * delta_y + sy * cp * cr * delta_z,
          cy * cp * delta_x + (-sy * cr + cy * sp * sr) * delta_y + (sy * sr + cy * sp * sr) * delta_z,
          cp * cr * delta_y - cp * sr * delta_z, -cp * delta_x - sp * sr * delta_y - sp * cr * delta_z, 0,
          tp * cr * delta_pitch - tp * sr * delta_yaw + 1,
          (1 + tp * tp) * sr * delta_pitch + (1 + tp + tp) * cr * delta_yaw, 0, -sr * delta_pitch - cr * delta_yaw, 1,
          0, (1 / cp) * (cr * delta_pitch - sr * delta_yaw),
          (1 / (cp * cp)) * (sr * sp * delta_pitch + sp * cr * delta_yaw), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }

    // Jacobian wrt vel_linear1
    if (jacobians[2])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[2]);
      jacobian << cy * cp * dt, (-sy * cr + cy * sp * sr) * dt, (sy * sr + cy * sp * cr) * dt, sy * cp * dt,
          (cy * cr + sy * sp * sr) * dt, (-cy * sr + sy * sp * sr) * dt, -sp * dt, cp * sr * dt, cp * cr * dt, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0;
    }

    // Jacobian wrt vel_angular1
    if (jacobians[3])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[3]);
      jacobian << 0, 0, 0, 0, 0, 0, 0, 0, 0, tp * sr * dt, tp * cr * dt, 0, 0, cr * dt, -sr * dt, 0, (sr / cp) * dt,
          (cr / cp) * dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0;
    }

    // Jacobian wrt acc_linear1
    if (jacobians[4])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[4]);
      jacobian << 0.5 * cy * cp * dt * dt, 0.5 * (-sy * cr + cy * sp * sr) * dt * dt,
          0.5 * (sy * sr + cy * sp * cr) * dt * dt, 0.5 * sy * cy * dt * dt, 0.5 * (cy * cr + sy * sp * sr) * dt * dt,
          0.5 * (-cy * sr + sy * sp * sr) * dt * dt - 0.5 * sp * dt * dt, 0, 5 * cp * sr * dt * dt,
          0.5 * cp * cr * dt * dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, dt, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }
    // Jacobian wrt acc_angular1
    if (jacobians[5])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[5]);
      jacobian << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5 * tp * sr * dt * dt, 0.5 * tp * cr * dt * dt, 0, 0.5 * cr * dt * dt,
          -0.5 * sr * dt * dt, 0, (sr * dt * dt) / (2 * cp), (cr * dt * dt) / (2 * cp), 0, 0, 0, 0, 0, 0, 0, 0, 0, dt,
          0, 0, 0, dt, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    }
  }
}

/**
 *
 *
 * @brief Given a state and time delta, predicts a new state
 *  @brief Given a state and time delta, predicts a new state
 * @param[in] position1_x - First X position
 * @param[in] position1_y - First Y position
 * @param[in] position1_z - First Y position
 * @param[in] orientation1_w - First orientation
 * @param[in] orientation1_x - First orientation
 * @param[in] orientation1_y - First orientation
 * @param[in] orientation1_z - First orientation
 * @param[in] vel_linear1_x - First X velocity
 * @param[in] vel_linear1_y - First Y velocity
 * @param[in] vel_linear1_z - First Y velocity
 * @param[in] vel_angular1_x - First yaw velocity
 * @param[in] vel_angular1_y - First yaw velocity
 * @param[in] vel_angular1_z - First yaw velocity
 * @param[in] acc_linear1_x - First X acceleration
 * @param[in] acc_linear1_y - First Y acceleration
 * @param[in] acc_linear1_z - First Y acceleration
 * @param[in] acc_angular1_x - First X acceleration
 * @param[in] acc_angular1_y - First Y acceleration
 * @param[in] acc_angular1_z - First Y acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2_x - Second X position
 * @param[out] position2_y - Second Y position
 * @param[out] position2_z - Second Y position
 *  @param[out] orientation2_w - Second orientation
 * @param[out] orientation2_x - Second orientation
 * @param[out] orientation2_y - Second orientation
 * @param[out] orientation2_z - Second orientation
 * @param[out] vel_linear2_x - Second X velocity
 * @param[out] vel_linear2_y - Second Y velocity
 * @param[out] vel_linear2_z - Second Y velocity
 * @param[out] vel_angular2_x - Second yaw velocity
 * @param[out] vel_angular2_y - Second yaw velocity
 * @param[out] vel_angular2_z - Second yaw velocity
 * @param[out] acc_linear2_x - Second X acceleration
 * @param[out] acc_linear2_y - Second Y acceleration
 * @param[out] acc_linear2_z - Second Y acceleration
 * @param[out] acc_angular2_x - Second X acceleration
 * @param[out] acc_angular2_y - Second Y acceleration
 * @param[out] acc_angular2_z - Second Y acceleration
 * @param[out] jacobians - Jacobians wrt the state
 */
inline void predict(const double position1_x, const double position1_y, const double position1_z,
                    const double orientation1_w, const double orientation1_x, const double orientation1_y,
                    const double orientation1_z, const double vel_linear1_x, const double vel_linear1_y,
                    const double vel_linear1_z, const double vel_angular1_x, const double vel_angular1_y,
                    const double vel_angular1_z, const double acc_linear1_x, const double acc_linear1_y,
                    const double acc_linear1_z, const double acc_angular1_x, const double acc_angular1_y,
                    const double acc_angular1_z, const double dt, double& position2_x, double& position2_y,
                    double& position2_z, double& orientation2_w, double& orientation2_x, double& orientation2_y,
                    double& orientation2_z, double& vel_linear2_x, double& vel_linear2_y, double& vel_linear2_z,
                    double& vel_angular2_x, double& vel_angular2_y, double& vel_angular2_z, double& acc_linear2_x,
                    double& acc_linear2_y, double& acc_linear2_z, double& acc_angular2_x, double& acc_angular2_y,
                    double& acc_angular2_z, double** jacobians)
{
  double orientation1_quat[4];
  double orientation1_euler[3];
  orientation1_quat[0] = orientation1_w;
  orientation1_quat[1] = orientation1_x;
  orientation1_quat[2] = orientation1_y;
  orientation1_quat[3] = orientation1_z;
  ceres::QuaternionToAngleAxis(orientation1_quat, orientation1_euler);

  const double orientation1_roll = orientation1_euler[0];
  const double orientation1_pitch = orientation1_euler[1];
  const double orientation1_yaw = orientation1_euler[2];

  double delta_x = vel_linear1_x * dt + 0.5 * acc_linear1_x * dt * dt;
  double delta_y = vel_linear1_y * dt + 0.5 * acc_linear1_y * dt * dt;
  double delta_z = vel_linear1_z * dt + 0.5 * acc_linear1_z * dt * dt;

  double delta_roll = vel_angular1_x * dt + 0.5 * acc_angular1_x * dt * dt;
  double delta_pitch = vel_angular1_y * dt + 0.5 * acc_angular1_y * dt * dt;
  double delta_yaw = vel_angular1_z * dt + 0.5 * acc_angular1_z * dt * dt;

  double sr = ceres::sin(orientation1_roll);
  double cr = ceres::cos(orientation1_roll);

  double sp = ceres::sin(orientation1_pitch);
  double cp = ceres::cos(orientation1_pitch);

  double cpi = 1 / cp;
  double tp = sp * cpi;

  double sy = ceres::sin(orientation1_yaw);
  double cy = ceres::cos(orientation1_yaw);

  position2_x =
      position1_x + (cy * cp) * delta_x + (cy * sp * sr - sy * cr) * delta_y + (cy * sp * cr + sy * sr) * delta_z;
  position2_y =
      position1_y + (sy * cp) * delta_x + (sy * sp * sr + cy * cr) * delta_y + (sy * sp * cr - cy * sr) * delta_z;
  position2_z = position1_z + (-sp) * delta_x + (cp * sr) * delta_y + (cp * cr) * delta_z;

  double orientation2_roll = orientation1_roll + delta_roll + (sr * tp) * delta_pitch + (cr * tp) * delta_yaw;
  double orientation2_pitch = orientation1_pitch + (cr)*delta_pitch + (-sr) * delta_yaw;
  double orientation2_yaw = orientation1_yaw + (sr * cpi) * delta_pitch + (cr * cpi) * delta_yaw;

  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_linear2_z = vel_linear1_z + acc_linear1_z * dt;

  vel_angular2_x = vel_angular1_x + acc_angular1_x * dt;
  vel_angular2_y = vel_angular1_y + acc_angular1_y * dt;
  vel_angular2_z = vel_angular1_z + acc_angular1_z * dt;

  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;
  acc_linear2_z = acc_linear1_z;

  acc_angular2_x = acc_angular1_x;
  acc_angular2_y = acc_angular1_y;
  acc_angular2_z = acc_angular1_z;

  double orientation2_euler[3];
  double orientation2_quat[4];

  fuse_core::wrapAngle(orientation2_roll);
  fuse_core::wrapAngle(orientation2_pitch);
  fuse_core::wrapAngle(orientation2_yaw);

  orientation2_euler[0] = orientation2_roll;
  orientation2_euler[1] = orientation2_pitch;
  orientation2_euler[2] = orientation2_yaw;

  ceres::AngleAxisToQuaternion(orientation2_euler, orientation2_quat);
  orientation2_w = orientation2_quat[0];
  orientation2_x = orientation2_quat[1];
  orientation2_y = orientation2_quat[2];
  orientation2_z = orientation2_quat[3];

  vel_linear2_x = vel_linear1_x + acc_linear1_x * dt;
  vel_linear2_y = vel_linear1_y + acc_linear1_y * dt;
  vel_linear2_z = vel_linear1_z + acc_linear1_z * dt;

  vel_angular2_x = vel_angular1_x + acc_angular1_x * dt;
  vel_angular2_y = vel_angular1_y + acc_angular1_y * dt;
  vel_angular2_z = vel_angular1_z + acc_angular1_z * dt;

  acc_linear2_x = acc_linear1_x;
  acc_linear2_y = acc_linear1_y;
  acc_linear2_z = acc_linear1_z;

  acc_angular2_x = acc_angular1_x;
  acc_angular2_y = acc_angular1_y;
  acc_angular2_z = acc_angular1_z;

  fuse_core::wrapAngle(vel_angular1_x);
  fuse_core::wrapAngle(vel_angular1_y);
  fuse_core::wrapAngle(vel_angular1_z);

  fuse_core::wrapAngle(acc_angular1_x);
  fuse_core::wrapAngle(acc_angular1_y);
  fuse_core::wrapAngle(acc_angular1_z);

  if (jacobians)
  {
    // Jacobian wrt position1
    if (jacobians[0])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[0]);
      jacobian << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }

    if (jacobians[1])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[1]);
      jacobian << (sy * sr + cy * sp * cr) * delta_y + (sy * sr - cy * sp * sr) * delta_z,
          (-cy * sp * delta_x) * (cy * cp * sr) * delta_y + (cy * cp * cr) * delta_z,
          -sy * cp * delta_x + (-cy * cr - sy * sp * sr) * delta_y + (cy * sr - sy * sp * cr) * delta_z,
          (-cy * sr + sy * sp * cr) * delta_y + (-cy * cr - sy * sp * sr) * delta_z,
          -sy * sp * delta_x + sy * cp * sr * delta_y + sy * cp * cr * delta_z,
          cy * cp * delta_x + (-sy * cr + cy * sp * sr) * delta_y + (sy * sr + cy * sp * sr) * delta_z,
          cp * cr * delta_y - cp * sr * delta_z, -cp * delta_x - sp * sr * delta_y - sp * cr * delta_z, 0,
          tp * cr * delta_pitch - tp * sr * delta_yaw + 1,
          (1 + tp * tp) * sr * delta_pitch + (1 + tp + tp) * cr * delta_yaw, 0, -1 * sr * delta_pitch - cr * delta_yaw,
          1, 0, (1 / cp) * (cr * delta_pitch - sr * delta_yaw),
          (1 / (cp * cp)) * (sr * sp * delta_pitch + sp * cr * delta_yaw), 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }

    // Jacobian wrt vel_linear1
    if (jacobians[2])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[2]);
      jacobian << cy * cp * dt, (-sy * cr + cy * sp * sr) * dt, (sy * sr + cy * sp * cr) * dt, sy * cp * dt,
          (cy * cr + sy * sp * sr) * dt, (-cy * sr + sy * sp * sr) * dt, -sp * dt, cp * sr * dt, cp * cr * dt, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0;
    }

    // Jacobian wrt vel_angular1
    if (jacobians[3])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[3]);
      jacobian << 0, 0, 0, 0, 0, 0, 0, 0, 0, tp * sr * dt, tp * cr * dt, 0, 0, cr * dt, -sr * dt, 0, (sr / cp) * dt,
          (cr / cp) * dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0;
    }

    // Jacobian wrt acc_linear1
    if (jacobians[4])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[4]);
      jacobian << 0.5 * cy * cp * dt * dt, 0.5 * (-sy * cr + cy * sp * sr) * dt * dt,
          0.5 * (sy * sr + cy * sp * cr) * dt * dt, 0.5 * sy * cy * dt * dt, 0.5 * (cy * cr + sy * sp * sr) * dt * dt,
          0.5 * (-cy * sr + sy * sp * sr) * dt * dt - 0.5 * sp * dt * dt, 0, 5 * cp * sr * dt * dt,
          0.5 * cp * cr * dt * dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, dt, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }
    // Jacobian wrt acc_angular1
    if (jacobians[5])
    {
      Eigen::Map<fuse_core::Matrix<double, 18, 3>> jacobian(jacobians[5]);
      jacobian << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5 * tp * sr * dt * dt, 0.5 * tp * cr * dt * dt, 0, 0.5 * cr * dt * dt,
          -0.5 * sr * dt * dt, 0, (sr * dt * dt) / (2 * cp), (cr * dt * dt) / (2 * cp), 0, 0, 0, 0, 0, 0, 0, 0, 0, dt,
          0, 0, 0, dt, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    }
  }
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] position1 - First position (array with x at index 0, y at index 1)
 * @param[in] orientation1 - First orientation
 * @param[in] vel_linear1 - First velocity (array with x at index 0, y at index 1)
 * @param[in] vel_angular1 - First yaw velocity
 * @param[in] acc_linear1 - First linear acceleration (array with x at index 0, y at index 1)
 * @param[in] acc_angular1 - First linear acceleration (array with x at index 0, y at index 1)
 * @param[in] dt - The time delta across which to predict the state
 * @param[out] position2 - Second position (array with x at index 0, y at index 1)
 * @param[out] orientation2 - Second orientation
 * @param[out] vel_linear2 - Second velocity (array with x at index 0, y at index 1)
 * @param[out] vel_angular2 - Second yaw velocity
 * @param[out] acc_linear2 - Second linear acceleration (array with x at index 0, y at index 1)
 * @param[out] acc_angular2 - Second linear acceleration (array with x at index 0, y at index 1)
 */
template <typename T>
inline void predict(const T* const position1, const T* const orientation1, const T* const vel_linear1,
                    const T* const vel_angular1, const T* const acc_linear1, const T* const acc_angular1, const T dt,
                    T* const position2, T* const orientation2, T* const vel_linear2, T* const vel_angular2,
                    T* const acc_linear2, T* const acc_angular2)
{
  predict(position1[0], position1[1], position1[2], orientation1[0], orientation1[1], orientation1[2], orientation1[3],
          vel_linear1[0], vel_linear1[1], vel_linear1[2], vel_angular1[0], vel_angular1[1], vel_angular1[2],
          acc_linear1[0], acc_linear1[1], acc_linear1[2], acc_angular1[0], acc_angular1[1], acc_angular1[2], dt,
          position2[0], position2[1], position2[2], orientation2[0], orientation2[1], orientation2[2], orientation2[3],
          vel_linear2[0], vel_linear2[1], vel_linear2[2], vel_angular2[0], vel_angular2[1], vel_angular2[2],
          acc_linear2[0], acc_linear2[1], acc_linear2[2], acc_angular2[0], acc_angular2[1], acc_angular2[2]);
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] pose1 - The first 3D pose
 * @param[in] vel_linear1 - The first linear velocity
 * @param[in] vel_angular1 - The first yaw velocity
 * @param[in] acc_linear1 - The first linear acceleration
 * @param[in] acc_angular1 - The first linear acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[in] pose2 - The second 3D pose
 * @param[in] vel_linear2 - The second linear velocity
 * @param[in] vel_angular2 - The second yaw velocity
 * @param[in] acc_linear2 - The second linear acceleration
 * @param[in] acc_angular2 - The second linear acceleration
 * @param[in] jacobian - The jacobian wrt the state
 */
inline void predict(const geometry_msgs::Pose& pose1, const geometry_msgs::Twist& vel_linear1,
                    const geometry_msgs::Twist& vel_angular1, const geometry_msgs::Accel& acc_linear1,
                    const geometry_msgs::Accel& acc_angular1, const double dt, geometry_msgs::Pose& pose2,
                    geometry_msgs::Twist& vel_linear2, geometry_msgs::Twist& vel_angular2,
                    geometry_msgs::Accel& acc_linear2, geometry_msgs::Accel& acc_angular2,
                    fuse_core::Matrix18d& jacobian)

{
  double pos_x_pred{};
  double pos_y_pred{};
  double pos_z_pred{};
  double quat_w_pred{};
  double quat_x_pred{};
  double quat_y_pred{};
  double quat_z_pred{};
  double vel_linear_x_pred{};
  double vel_linear_y_pred{};
  double vel_linear_z_pred{};
  double vel_angular_x_pred{};
  double vel_angular_y_pred{};
  double vel_angular_z_pred{};
  double acc_linear_x_pred{};
  double acc_linear_y_pred{};
  double acc_linear_z_pred{};
  double acc_angular_x_pred{};
  double acc_angular_y_pred{};
  double acc_angular_z_pred{};

  // fuse_core::Matrix18d is Eigen::RowMajor, so we cannot use pointers to the columns where each parameter block
  // starts. Instead, we need to create a vector of Eigen::RowMajor matrices per parameter block and later reconstruct
  // the fuse_core::Matrix18d with the full jacobian. The parameter blocks have the following sizes: {position1: 3,
  // orientation1: 3, vel_linear1: 3, vel_angular1: 3, accel_lineaer1: 3 acc_angular1: 3}
  static constexpr size_t num_residuals{ 18 };
  static constexpr size_t num_parameter_blocks{ 6 };
  static const std::array<size_t, num_parameter_blocks> block_sizes = { 3, 4, 3, 3, 3, 3 };

  std::array<fuse_core::MatrixXd, num_parameter_blocks> J;
  std::array<double*, num_parameter_blocks> jacobians;

  for (size_t i = 0; i < num_parameter_blocks; ++i)
  {
    J[i].resize(num_residuals, block_sizes[i]);
    jacobians[i] = J[i].data();
  }

  predict(pose1.position.x, pose1.position.y, pose1.position.z, pose1.orientation.w, pose1.orientation.x,
          pose1.orientation.y, pose1.orientation.z, vel_linear1.linear.x, vel_linear1.linear.y, vel_linear1.linear.z,
          vel_angular1.linear.x, vel_angular1.linear.y, vel_angular1.linear.z, acc_linear1.linear.x,
          acc_linear1.linear.y, acc_linear1.linear.z, acc_angular1.linear.x, acc_angular1.linear.y,
          acc_angular1.linear.z, dt, pos_x_pred, pos_y_pred, pos_z_pred, quat_w_pred, quat_x_pred, quat_y_pred,
          quat_z_pred, vel_linear_x_pred, vel_linear_y_pred, vel_linear_z_pred, vel_angular_x_pred, vel_angular_y_pred,
          vel_angular_z_pred, acc_linear_x_pred, acc_linear_y_pred, acc_linear_z_pred, acc_angular_x_pred,
          acc_angular_y_pred, acc_angular_z_pred, jacobians.data());
  jacobian << J[0], J[1], J[2], J[3], J[4], J[5], J[6], J[7], J[8], J[9], J[10], J[11], J[12], J[13], J[14], J[15],
      J[16], J[17];
  pose2.position.x = pos_x_pred;
  pose2.position.y = pos_y_pred;
  pose2.position.z = pos_z_pred;
  pose2.orientation.w = quat_w_pred;
  pose2.orientation.x = quat_x_pred;
  pose2.orientation.y = quat_y_pred;
  pose2.orientation.z = quat_z_pred;
  vel_linear2.linear.x = vel_linear_x_pred;
  vel_linear2.linear.y = vel_linear_y_pred;
  vel_linear2.linear.z = vel_linear_z_pred;
  vel_angular2.linear.x = vel_angular_x_pred;
  vel_angular2.linear.y = vel_angular_y_pred;
  vel_angular2.linear.z = vel_angular_z_pred;
  acc_linear2.linear.x = acc_linear_x_pred;
  acc_linear2.linear.y = acc_linear_y_pred;
  acc_linear2.linear.z = acc_linear_z_pred;
  acc_angular2.linear.x = acc_angular_x_pred;
  acc_angular2.linear.y = acc_angular_y_pred;
  acc_angular2.linear.z = acc_angular_z_pred;
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] pose1 - The first 3D pose
 * @param[in] vel_linear_1 - The first linear velocity
 * @param[in] vel_angular1 - The first yaw velocity
 * @param[in] acc_linear1 - The first linear acceleration
 * @param[in] acc_angular1 - The first angular acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[in] pose2 - The second 3D pose
 * @param[in] vel_linear_2 - The second linear velocity
 * @param[in] vel_angular_2 - The second linear velocity
 * @param[in] acc_linear2 - The second linear acceleration
 * @param[in] acc_angular2 - The second linear acceleration
 */
inline void predict(const geometry_msgs::Pose& pose1, const geometry_msgs::Twist& vel_linear1,
                    const geometry_msgs::Twist& vel_angular1, const geometry_msgs::Accel& acc_linear1,
                    const geometry_msgs::Accel& acc_angular1, const double dt, geometry_msgs::Pose& pose2,
                    geometry_msgs::Twist& vel_linear2, geometry_msgs::Twist& vel_angular2,
                    geometry_msgs::Accel& acc_linear2, geometry_msgs::Accel& acc_angular2)
{
  double pos_x_pred{};
  double pos_y_pred{};
  double pos_z_pred{};
  double quat_w_pred{};
  double quat_x_pred{};
  double quat_y_pred{};
  double quat_z_pred{};
  double vel_linear_x_pred{};
  double vel_linear_y_pred{};
  double vel_linear_z_pred{};
  double vel_angular_x_pred{};
  double vel_angular_y_pred{};
  double vel_angular_z_pred{};
  double acc_linear_x_pred{};
  double acc_linear_y_pred{};
  double acc_linear_z_pred{};
  double acc_angular_x_pred{};
  double acc_angular_y_pred{};
  double acc_angular_z_pred{};

  predict(pose1.position.x, pose1.position.y, pose1.position.z, pose1.orientation.w, pose1.orientation.x,
          pose1.orientation.y, pose1.orientation.z, vel_linear1.linear.x, vel_linear1.linear.y, vel_linear1.linear.z,
          vel_angular1.angular.x, vel_angular1.angular.y, vel_angular1.angular.z, acc_linear1.linear.x,
          acc_linear1.linear.y, acc_linear1.linear.z, acc_angular1.angular.x, acc_angular1.angular.y,
          acc_angular1.angular.z, dt, pos_x_pred, pos_y_pred, pos_z_pred, quat_w_pred, quat_x_pred, quat_y_pred,
          quat_z_pred, vel_linear_x_pred, vel_linear_y_pred, vel_linear_z_pred, vel_angular_x_pred, vel_angular_y_pred,
          vel_angular_z_pred, acc_linear_x_pred, acc_linear_y_pred, acc_linear_z_pred, acc_angular_x_pred,
          acc_angular_y_pred, acc_angular_z_pred);

  pose2.position.x = pos_x_pred;
  pose2.position.y = pos_y_pred;
  pose2.position.z = pos_z_pred;
  pose2.orientation.w = quat_w_pred;
  pose2.orientation.x = quat_x_pred;
  pose2.orientation.y = quat_y_pred;
  pose2.orientation.z = quat_z_pred;
  vel_linear2.linear.x = vel_linear_x_pred;
  vel_linear2.linear.y = vel_linear_y_pred;
  vel_linear2.linear.z = vel_linear_z_pred;
  vel_angular2.angular.x = vel_angular_x_pred;
  vel_angular2.angular.y = vel_angular_y_pred;
  vel_angular2.angular.z = vel_angular_z_pred;
  acc_linear2.linear.x = acc_linear_x_pred;
  acc_linear2.linear.y = acc_linear_y_pred;
  acc_linear2.linear.z = acc_linear_z_pred;
  acc_angular2.angular.x = acc_angular_x_pred;
  acc_angular2.angular.y = acc_angular_y_pred;
  acc_angular2.angular.z = acc_angular_z_pred;
}

/**
 * @brief Given a state and time delta, predicts a new state
 * @param[in] pose1 - The first 3D pose
 * @param[in] vel_linear_1 - The first linear velocity
 * @param[in] vel_angular1 - The first yaw velocity
 * @param[in] acc_linear1 - The first linear acceleration
 * @param[in] acc_angular1 - The first angular acceleration
 * @param[in] dt - The time delta across which to predict the state
 * @param[in] pose2 - The second 3D pose
 * @param[in] vel_linear_2 - The second linear velocity
 * @param[in] vel_angular_2 - The second linear velocity
 * @param[in] acc_linear2 - The second linear acceleration
 * @param[in] acc_angular2 - The second linear acceleration
 */
inline void predict(const tf2::Transform& pose1, const tf2::Vector3& vel_linear1, const tf2::Vector3& vel_angular1,
                    const tf2::Vector3& acc_linear1, const tf2::Vector3& acc_angular1, const double dt,
                    tf2::Transform& pose2, tf2::Vector3& vel_linear2, tf2::Vector3& vel_angular2,
                    tf2::Vector3& acc_linear2, tf2::Vector3& acc_angular2, fuse_core::Matrix18d& jacobian)
{
  double pos_x_pred{};
  double pos_y_pred{};
  double pos_z_pred{};
  double quat_w_pred{};
  double quat_x_pred{};
  double quat_y_pred{};
  double quat_z_pred{};
  double vel_linear_x_pred{};
  double vel_linear_y_pred{};
  double vel_linear_z_pred{};
  double vel_angular_x_pred{};
  double vel_angular_y_pred{};
  double vel_angular_z_pred{};
  double acc_linear_x_pred{};
  double acc_linear_y_pred{};
  double acc_linear_z_pred{};
  double acc_angular_x_pred{};
  double acc_angular_y_pred{};
  double acc_angular_z_pred{};

  tf2::Quaternion tempQ;

  // fuse_core::Matrix18d is Eigen::RowMajor, so we cannot use pointers to the columns where each parameter block
  // starts. Instead, we need to create a vector of Eigen::RowMajor matrices per parameter block and later reconstruct
  // the fuse_core::Matrix18d with the full jacobian. The parameter blocks have the following sizes: {position1: 3,
  // orientation1: 3, vel_linear1: 3, vel_angular1: 3, accel_lineaer1: 3 acc_angular1: 3}
  static constexpr size_t num_residuals{ 18 };
  static constexpr size_t num_parameter_blocks{ 6 };
  static const std::array<size_t, num_parameter_blocks> block_sizes = { 3, 4, 3, 3, 3, 3 };

  std::array<fuse_core::MatrixXd, num_parameter_blocks> J;
  std::array<double*, num_parameter_blocks> jacobians;

  for (size_t i = 0; i < num_parameter_blocks; ++i)
  {
    J[i].resize(num_residuals, block_sizes[i]);
    jacobians[i] = J[i].data();
  }

  predict(pose1.getOrigin().getX(), pose1.getOrigin().getY(), pose1.getOrigin().getZ(), pose1.getRotation().getW(),
          pose1.getRotation().getX(), pose1.getRotation().getY(), pose1.getRotation().getZ(), vel_linear1.getX(),
          vel_linear1.getY(), vel_linear1.getX(), vel_angular1.getX(), vel_angular1.getY(), vel_angular1.getZ(),
          acc_linear1.getX(), acc_linear1.getY(), acc_linear1.getZ(), acc_angular1.getX(), acc_angular1.getY(),
          acc_angular1.getZ(), dt, pos_x_pred, pos_y_pred, pos_z_pred, quat_w_pred, quat_x_pred, quat_y_pred,
          quat_z_pred, vel_linear_x_pred, vel_linear_y_pred, vel_linear_z_pred, vel_angular_x_pred, vel_angular_y_pred,
          vel_angular_z_pred, acc_linear_x_pred, acc_linear_y_pred, acc_linear_z_pred, acc_angular_x_pred,
          acc_angular_y_pred, acc_angular_z_pred, jacobians.data());

  jacobian << J[0], J[1], J[2], J[3], J[4], J[5], J[6], J[7], J[8], J[9], J[10], J[11], J[12], J[13], J[14], J[15],
      J[16], J[17];
  pose2.setOrigin(tf2::Vector3(pos_x_pred, pos_y_pred, pos_z_pred));
  pose2.setRotation(tf2::Quaternion(quat_x_pred, quat_y_pred, quat_z_pred, quat_w_pred));
  vel_linear2.setX(vel_linear_x_pred);
  vel_linear2.setY(vel_linear_y_pred);
  vel_linear2.setZ(vel_linear_z_pred);
  vel_angular2.setX(vel_angular_x_pred);
  vel_angular2.setY(vel_angular_y_pred);
  vel_angular2.setZ(vel_angular_z_pred);
  acc_linear2.setX(acc_linear_x_pred);
  acc_linear2.setY(acc_linear_y_pred);
  acc_linear2.setZ(acc_linear_z_pred);
  acc_angular2.setX(acc_angular_x_pred);
  acc_angular2.setY(acc_angular_y_pred);
  acc_angular2.setZ(acc_angular_z_pred);
}
}  // namespace fuse_models

#endif  // FUSE_MODELS_SKID_STEER_3D_PREDICT_H
