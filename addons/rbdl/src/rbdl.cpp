/** \file rbdl.cpp
 * \brief RBDL environment source file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-04-01
 *
 * \copyright \verbatim
 * Copyright (c) 2015, Wouter Caarls
 * All rights reserved.
 *
 * This file is part of GRL, the Generic Reinforcement Learning library.
 *
 * GRL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * \endverbatim
 */

#include <rbdl/rbdl.h>
#include <rbdl/addons/luamodel/luamodel.h>

#include <grl/environments/rbdl.h>

using namespace grl;

REGISTER_CONFIGURABLE(RBDLDynamics)

void RBDLDynamics::request(ConfigurationRequest *config)
{
  config->push_back(CRP("file", "RBDL Lua model file", file_, CRP::Configuration));
}

void RBDLDynamics::configure(Configuration &config)
{
  file_ = config["file"].str();
  
  model_ = new RigidBodyDynamics::Model();

  INFO("Loading model from " << file_);
  if (!RigidBodyDynamics::Addons::LuaModelReadFromFile(file_.c_str(), model_))
  {
    ERROR("Error loading model " << file_);
    throw bad_param("dynamics/rbdl:file");
  }
  
  NOTICE("Loaded RBDL model with " << model_->dof_count << " degrees of freedom");
}

void RBDLDynamics::reconfigure(const Configuration &config)
{
}

RBDLDynamics *RBDLDynamics::clone() const
{
  return NULL;
}

void RBDLDynamics::eom(const Vector &state, const Vector &action, Vector *xd) const
{
  size_t dim = model_->dof_count;

  RigidBodyDynamics::Math::VectorNd u = RigidBodyDynamics::Math::VectorNd::Zero(dim);
  RigidBodyDynamics::Math::VectorNd q = RigidBodyDynamics::Math::VectorNd::Zero(dim);
  RigidBodyDynamics::Math::VectorNd qd = RigidBodyDynamics::Math::VectorNd::Zero(dim);
  RigidBodyDynamics::Math::VectorNd qdd = RigidBodyDynamics::Math::VectorNd::Zero(dim);

  for (size_t ii=0; ii < dim; ++ii)
  {
    u[ii] = action[ii];
    q[ii] = state[ii];
    qd[ii] = state[ii + dim];
  }

  RigidBodyDynamics::ForwardDynamics(*model_, q, qd, u, qdd);

  Vector res(2*dim);

  for (size_t ii=0; ii < dim; ++ii)
  {
    (*xd)[ii] = qd[ii];
    (*xd)[ii + dim] = qdd[ii];
  }
}
