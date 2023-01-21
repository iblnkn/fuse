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
#include <fuse_constraints/normal_prior_orientation_3d.h>

#include <ceres/rotation.h>
#include <fuse_core/manifold.h>

namespace fuse_constraints
{
NormalPriorOrientation3D::NormalPriorOrientation3D(const fuse_core::MatrixXd& A, const fuse_core::Vector4d& b)
  : A_(A), b_(b)
{
  CHECK_GT(A_.rows(), 0);
  CHECK_EQ(A_.cols(), 3);
  set_num_residuals(A_.rows());
}

bool NormalPriorOrientation3D::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
  // Compute the delta quaternion
  double variable[4] = { parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3] };

  double observation_inverse[4] = { b_(0), -b_(1), -b_(2), -b_(3) };

  double difference[4];
  ceres::QuaternionProduct(observation_inverse, variable, difference);
  ceres::QuaternionToAngleAxis(difference, residuals);

  // Scale the residuals by the square root information matrix to account for the measurement uncertainty.
  Eigen::Map<Eigen::Matrix<double, 3, 1>> residuals_map(residuals);
  residuals_map.applyOnTheLeft(A_.template cast<double>());
  jacobians[0][0] = 1;
  jacobians[0][1] = 0;
  jacobians[0][2] = 0;
  jacobians[1][0] = 0;
  jacobians[1][1] = 1;
  jacobians[1][2] = 0;
  jacobians[2][0] = 0;
  jacobians[2][1] = 0;
  jacobians[2][2] = 1;

  return true;
}

}  // namespace fuse_constraints
