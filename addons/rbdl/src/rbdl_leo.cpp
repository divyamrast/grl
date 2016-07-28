/** \file rbdl_leo.cpp
 * \brief RBDL file for C++ description of Leo task.
 *
 * \author    Ivan Koryakovskiy <i.koryakovskiy@tudelft.nl>
 * \date      2016-06-30
 *
 * \copyright \verbatim
 * Copyright (c) 2016, Ivan Koryakovskiy
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

#include <sys/stat.h>
#include <libgen.h>

#include <rbdl/rbdl.h>
#include <rbdl/addons/luamodel/luamodel.h>

#include <grl/lua_utils.h>
#include <grl/environments/rbdl_leo.h>
//#include <grl/environments/leohelper.h>

using namespace grl;

REGISTER_CONFIGURABLE(LeoRBDLDynamics)
REGISTER_CONFIGURABLE(LeoSquatTask)


void LeoRBDLDynamics::finalize(Vector &state)
{
  RBDLState *rbdl = rbdl_state_.instance();

  size_t dim = rbdl->model->dof_count;

  if (state.size() != 2*dim+1)
    throw Exception("dynamics/rbdl is incompatible with specified task");

  leo_helper_.model = rbdl->model;
  leo_helper_.calcMomentum();

  RigidBodyDynamics::Math::Vector3d com = leo_helper_.calcCenterOfMass();

  Vector augmented_state;
  augmented_state.resize(state.size()+2);
  augmented_state << state, com[0], com[2];
  state = augmented_state;
}

//-----------------------------------------------------------

void LeoSquatTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
  config->push_back(CRP("observation_min", "vector.observation_min", "Lower limit on observations", CRP::System));
  config->push_back(CRP("observation_max", "vector.observation_max", "Upper limit on observations", CRP::System));
  config->push_back(CRP("action_min", "vector.action_min", "Lower limit on actions", CRP::System));
  config->push_back(CRP("action_max", "vector.action_max", "Upper limit on actions", CRP::System));
}

void LeoSquatTask::configure(Configuration &config)
{
  timeout_ = config["timeout"];

  // augment observations with time
  Vector observation_min = config["observation_min"].v();
  Vector observation_max = config["observation_max"].v();
  grl_assert(observation_min.size() == observation_max.size());
  observation_min_.resize(observation_min.size()+1);
  observation_max_.resize(observation_max.size()+1);
  observation_min_ << observation_min, 0;
  observation_max_ << observation_max, timeout_;

  action_min_ = config["action_min"].v();
  action_max_ = config["action_max"].v();
  grl_assert(action_min_.size() == action_max_.size());

  config.set("reward_min", VectorConstructor(-1000));
  config.set("reward_max", VectorConstructor( 1000));

  observation_dims_ = observation_min_.size();
  action_dims_ = action_min_.size();
  config.set("observation_dims", observation_dims_);
  config.set("action_dims", action_dims_);

  //Vector v_obs_min, v_obs_max;
  //config.set("observation_dims", observation_dims_);
  //std::vector<double> obs_min = {-M_PI, -M_PI, -M_PI, -M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI,   0};
  //std::vector<double> obs_max = { M_PI,  M_PI,  M_PI,  M_PI,  10*M_PI,  10*M_PI,  10*M_PI,  10*M_PI, 100}; // last is time
  //toVector(obs_min, v_obs_min);
  //toVector(obs_max, v_obs_max);
  //config.set("observation_min", v_obs_min);
  //config.set("observation_max", v_obs_max);

  //config.set("action_min", VectorConstructor(-10.7, -10.7, -10.7, -10.7));
  //config.set("action_max", VectorConstructor( 10.7,  10.7,  10.7,  10.7));
}

void LeoSquatTask::reconfigure(const Configuration &config)
{
}
 
LeoSquatTask *LeoSquatTask::clone() const
{
  return new LeoSquatTask(*this);
}
 
void LeoSquatTask::start(int test, Vector *state) const
{
  *state = ConstantVector(observation_dims_, 0);
  /* // fb_sl, muscod
  *state <<
         1.0586571916803691E+00,
        -2.1266836153365212E+00,
         1.0680264236561250E+00,
        -2.5999999999984957E-01,
        -0.0,
        -0.0,
        -0.0,
        -0.0,
         0.0; // time
  */
  /* // fb_sl, leosim
  *state <<
         1.07106,
        -2.15961,
         1.06803,
        -0.2618,
         0.23794,
        -0.609729,
         5.3124e-05,
        -1.18169e-06,
         0.0; // time
   */

  // ff_sl
  *state <<
         2.9285671132918734E-02,
         2.7999999999998426E-01,
         1.5000000000041891E-01,
        -2.5999999999879975E-01,
         1.2544763628954583E+00,
        -2.1096010065342394E+00,
         1.0051246436392001E+00,
         0.0, // qdots
         0.0, // qdots
         0.0, // qdots
         0.0, // qdots
         0.0, // qdots
         0.0, // qdots
         0.0, // qdots
         0.0; // time;
}

int LeoSquatTask::failed(const Vector &state) const
{
  return 0;
}

void LeoSquatTask::observe(const Vector &state, Vector *obs, int *terminal) const
{
  *obs = state.block(0, 0, 1, observation_dims_-1); // exclude time from observations!

  if (state[observation_dims_-1] >= timeout_)
    *terminal = 1;
  else if (failed(state))
    *terminal = 2;
  else
    *terminal = 0;
}

void LeoSquatTask::evaluate(const Vector &state, const Vector &action, const Vector &next, double *reward) const
{
  *reward = 0;
}
 
bool LeoSquatTask::invert(const Vector &obs, Vector *state) const
{
  return true;
}

Matrix LeoSquatTask::rewardHessian(const Vector &state, const Vector &action) const
{
  Matrix hessian;
  return hessian;
}

