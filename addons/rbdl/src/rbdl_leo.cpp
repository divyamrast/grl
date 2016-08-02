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

enum RbdlLeoState
{
  rlsAnkleAngle,
  rlsKneeAngle,
  rlsHipAngle,
  rlsArmAngle,

  rlsDofDim = rlsArmAngle + 1,

  rlsAnkleAngleRate = rlsDofDim,
  rlsKneeAngleRate,
  rlsHipAngleRate,
  rlsArmAngleRate,

  rlsTime,
  rlsRefRootHeight,

  rlsLeftTipX,
  rlsLeftTipY,
  rlsLeftTipZ,

  rlsLeftHeelX,
  rlsLeftHeelY,
  rlsLeftHeelZ,

  rlsRootX,
  rlsRootY,
  rlsRootZ,

  rlsMass,

  rlsComX,
  rlsComY,
  rlsComZ,

  rlsComVelocityX,
  rlsComVelocityY,
  rlsComVelocityZ,

  rlsAngularMomentumX,
  rlsAngularMomentumY,
  rlsAngularMomentumZ,
  rlsStateDim = rlsAngularMomentumZ + 1
};

/*
void LeoRBDLDynamics::finalize(Vector &state)
{
  RBDLState *rbdl = rbdl_state_.instance();

  size_t dim = rbdl->model->dof_count;

  if (state.size() != 2*dim+1)
    throw Exception("dynamics/rbdl_leo is incompatible with specified task");

  leo_helper_.model = rbdl->model;
  leo_helper_.calcMomentum();

  RigidBodyDynamics::Math::Vector3d com = leo_helper_.calcCenterOfMass();

  Vector augmented_state;
  augmented_state.resize(state.size()+2);
  augmented_state << state, com[0], com[2];
  state = augmented_state;

}
*/

//-----------------------------------------------------------

void LeoSquatTask::request(ConfigurationRequest *config)
{
  Task::request(config);
  config->push_back(CRP("timeout", "double.timeout", "Task timeout", timeout_, CRP::System, 0.0, DBL_MAX));
}

void LeoSquatTask::configure(Configuration &config)
{
  timeout_ = config["timeout"];

  action_dims_ = 4;

  Vector v_obs_min, v_obs_max;
  config.set("observation_dims", 2*rlsDofDim + 1); // 2*dof + time
  std::vector<double> obs_min = {-M_PI, -M_PI, -M_PI, -M_PI, -10*M_PI, -10*M_PI, -10*M_PI, -10*M_PI, 0};
  std::vector<double> obs_max = { M_PI,  M_PI,  M_PI,  M_PI,  10*M_PI,  10*M_PI,  10*M_PI,  10*M_PI, 1};
  toVector(obs_min, v_obs_min);
  toVector(obs_max, v_obs_max);
  config.set("observation_min", v_obs_min);
  config.set("observation_max", v_obs_max);
  config.set("action_dims", action_dims_);
  config.set("action_min", VectorConstructor(-10.7, -10.7, -10.7, -10.7));
  config.set("action_max", VectorConstructor( 10.7,  10.7,  10.7,  10.7));
  config.set("reward_min", VectorConstructor(-1000));
  config.set("reward_max", VectorConstructor( 1000));
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
  *state = ConstantVector(rlsStateDim, 0);
  *state <<
         1.0586571916803691E+00,
        -2.1266836153365212E+00,
         1.0680264236561250E+00,
        -2.5999999999984957E-01,
        -0.0,
        -0.0,
        -0.0,
        -0.0, // end of rlsDofDim
         ConstantVector(rlsStateDim-2*rlsDofDim, 0); // initialize the rest to zero
}

int LeoSquatTask::failed(const Vector &state) const
{
  double torsoAngle = state[rlsAnkleAngle] + state[rlsKneeAngle] + state[rlsHipAngle];
  if ((torsoAngle < -1.0) || (torsoAngle > 1.0))
    return 1;
  else
    return 0;
}

void LeoSquatTask::observe(const Vector &state, Vector *obs, int *terminal) const
{
  grl_assert(state.size() == rlsStateDim);

  bool reduced = false;

  if (!reduced)
  {
    obs->resize(2*rlsDofDim + 1);
   *obs << state.block(0, 0, 1, 2*rlsDofDim), state[rlsRefRootHeight];
  }
  else
  {
    std::cout << state << std::endl;

    std::cout << state.block(0, 0, 1, 3) << std::endl; // angle
    std::cout << state.block(0, 4, 1, 3) << std::endl; // angle rate
    std::cout << state[8] << std::endl; // direction indicator

    obs->resize(state.size()-3);
    (*obs) << state.block(0, 0, 1, 3), state.block(0, 4, 1, 3), state[8];
  }

  if (state[rlsTime] >= timeout_)
    *terminal = 1;
  else if (failed(state))
    *terminal = 2;
  else
    *terminal = 0;
}

void LeoSquatTask::evaluate(const Vector &state, const Vector &action, const Vector &next, double *reward) const
{
  grl_assert(next.size() == rlsStateDim);

  if (failed(next))
  {
    *reward = -1000;
    return;
  }

  *reward = 0;

  // calculate support center from feet positions
  double suppport_center = 0.5 * (next[rlsLeftTipX] + next[rlsLeftHeelX]);

  // track: || root_z - h_ref ||_2^2
  *reward +=  pow(100.0 * (next[rlsRootZ] - next[rlsRefRootHeight]), 2);

  // track: || com_x,y - support center_x,y ||_2^2
  *reward +=  pow( 10.00 * (next[rlsComX] - suppport_center), 2);

  *reward +=  pow( 10.00 * next[rlsComVelocityX], 2);
  *reward +=  pow( 10.00 * next[rlsComVelocityZ], 2);

  *reward +=  pow( 10.00 * next[rlsAngularMomentumY], 2);

  // NOTE: sum of lower body angles is equal to angle between ground slope
  //       and torso. Minimizing deviation from zero keeps torso upright
  //       during motion execution.
  *reward += pow(10.00 * ( next[rlsAnkleAngle] + next[rlsKneeAngle] + next[rlsHipAngle] - (0.15) ), 2); // desired torso angle

  // regularize: || q - q_desired ||_2^2
  *reward += pow(1.0 * (next[rlsArmAngle]         - (-0.26)), 2); // arm

  // negate
  *reward = - *reward;
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

