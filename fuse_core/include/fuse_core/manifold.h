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
#ifndef FUSE_CORE_MANIFOLD_H
#define FUSE_CORE_MANIFOLD_H

#include <fuse_core/macros.h>
#include <fuse_core/serialization.h>

#include <boost/serialization/access.hpp>
#include <ceres/manifold.h>

namespace fuse_core
{
/**
 * @brief The Manifold interface definition.
 *
 * This class extends the Ceres Manifold class, adding the additional requirement of a
 * \p Minus() method, the conceptual inverse of the already required \p Plus() method.
 *
 * If Plus(x, delta) -> y, then Minus(x2, x1) -> delta
 *
 * See the Ceres documentation for more details. http://ceres-solver.org/nnls_modeling.html#localparameterization
 */
class Manifold : public ceres::Manifold
{
public:
  SMART_PTR_ALIASES_ONLY(Manifold);

  /**
   * @brief Generalization of the subtraction operation
   *
   * Minus(x2, x1) -> delta
   *
   * with the conditions that:
   *  - Minus(x, x) -> 0
   *  - if Plus(x, delta) -> y, then Minus(x2, x1) -> delta
   *
   * @param[in]  x2    The value of the first variable, of size \p AmbientSize()
   * @param[in]  x1    The value of the second variable, of size \p AmbientSize()
   * @param[out] delta The difference between the second variable and the first, of size \p TangentSize()
   * @return True if successful, false otherwise
   */
  virtual bool Minus(const double* x2, const double* x1, double* delta) const = 0;

  /**
   * @brief The jacobian of Minus(x2, x1) w.r.t y at x == y == x
   *
   * @param[in]  x        The value used to evaluate the Jacobian, of size \p AmbientSize()
   * @param[out] jacobian The first-order derivative in row-major order, of size \p TangentSize() x \p AmbientSize()
   * @return True if successful, false otherwise
   */
  virtual bool MinusJacobian(const double* x, double* jacobian) const = 0;

private:
  // Allow Boost Serialization access to private methods
  friend class boost::serialization::access;

  /**
   * @brief The Boost Serialize method that serializes all of the data members in to/out of the archive
   *
   * @param[in/out] archive - The archive object that holds the serialized class members
   * @param[in] version - The version of the archive being read/written. Generally unused.
   */
  template <class Archive>
  void serialize(Archive& /* archive */, const unsigned int /* version */)
  {
  }
};

}  // namespace fuse_core

#endif  // FUSE_CORE_MANIFOLD_H
