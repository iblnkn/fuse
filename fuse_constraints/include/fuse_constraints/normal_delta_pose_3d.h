/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2020, Locus Robotics
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
#ifndef FUSE_CONSTRAINTS_NORMAL_DELTA_POSE_3D_H
#define FUSE_CONSTRAINTS_NORMAL_DELTA_POSE_3D_H

#include <fuse_core/eigen.h>

#include <ceres/sized_cost_function.h>

namespace fuse_constraints
{
// TODO(iblankenau): Update this brief when implementing 3D cost function.
/**
 * @brief Implements a cost function that models a difference between pose variables.
 *
 */
class NormalDeltaPose3D : public ceres::SizedCostFunction<ceres::DYNAMIC, 3, 4, 3, 4>
{
public:
  /**
   * @brief Constructor
   *
   * The residual weighting matrix can vary in size, as this cost functor can be used to compute costs for partial
   * vectors. The number of rows of A will be the number of dimensions for which you want to compute the error, and the
   * number of columns in A will be fixed at 6. For example, if we just want to use the values of x and yaw, then \p A
   * will be of size 2x6.
   *
   * @param[in] A The residual weighting matrix, most likely the square root information matrix in order (x, y, yaw)
   * @param[in] b The exposed pose difference in order (x, y, z, roll, pitch, yaw)
   */
  NormalDeltaPose3D(const fuse_core::MatrixXd& A, const fuse_core::Vector7d& b);

  /**
   * @brief Compute the cost values/residuals, and optionally the Jacobians, using the provided variable/parameter
   * values
   */
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

private:
  fuse_core::MatrixXd A_;  //!< The residual weighting matrix, most likely the square root information matrix
  fuse_core::Vector7d b_;  //!< The measured difference between variable x0 and variable x1
};

}  // namespace fuse_constraints

#endif  // FUSE_CONSTRAINTS_NORMAL_DELTA_POSE_3D_H
