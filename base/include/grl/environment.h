/** \file environment.h
 * \brief Generic and basic environment definitions.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-01-22
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

#ifndef GRL_ENVIRONMENT_H_
#define GRL_ENVIRONMENT_H_

#include <grl/configurable.h>
#include <grl/state.h>
#include <grl/policy.h>

namespace grl
{

/// Real or simulated world.
class Environment : public Configurable
{
  public:
    virtual ~Environment() { }
    virtual Environment *clone() const = 0;

    virtual void start(int test, Vector *obs) = 0;
    virtual void step(const Vector &action, Vector *obs, double *reward, int *terminal) = 0;
};

/// Random-access transition model (works on states instead of observations).
class Model : public Configurable
{
  public:
    virtual ~Model() { }
    virtual Model *clone() const = 0;

    virtual void step(const Vector &state, const Vector &action, Vector *next) const = 0;
};

class Task : public Configurable
{
  public:
    virtual ~Task() { }
    virtual Task *clone() const = 0;

  public:
    virtual void request(ConfigurationRequest *config)
    {
      config->push_back(CRP("observation_dims", "int.observation_dims", "Number of observation dimensions", CRP::Provided));
      config->push_back(CRP("observation_min", "vector.observation_min", "Lower limit on observations", CRP::Provided));
      config->push_back(CRP("observation_max", "vector.observation_max", "Upper limit on observations", CRP::Provided));
      config->push_back(CRP("action_dims", "int.action_dims", "Number of action dimensions", CRP::Provided));
      config->push_back(CRP("action_min", "vector.action_min", "Lower limit on actions", CRP::Provided));
      config->push_back(CRP("action_max", "vector.action_max", "Upper limit on actions", CRP::Provided));
      config->push_back(CRP("reward_min", "double.reward_min", "Lower limit on immediate reward", CRP::Provided));
      config->push_back(CRP("reward_max", "double.reward_max", "Upper limit on immediate reward", CRP::Provided));
    }    
  
    virtual void start(int test, Vector *state) const = 0;
    virtual void observe(const Vector &state, Vector *obs, int *terminal) const = 0;
    virtual void evaluate(const Vector &state, const Vector &action, const Vector &next, double *reward) const = 0;
    virtual bool invert(const Vector &obs, Vector *state) const { return false; }
};

/// Environment that uses a transition model internally.
class ModeledEnvironment : public Environment
{
  public:
    TYPEINFO("environment/modeled")

  public:
    Model *model_;
    Task *task_;
    Vector state_;
    State *state_obj_;

  public:
    ModeledEnvironment() : model_(NULL), task_(NULL), state_obj_(NULL) { }
  
    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);
    
    // From Environment
    virtual ModeledEnvironment *clone() const;
    virtual void start(int test, Vector *obs);
    virtual void step(const Vector &action, Vector *obs, double *reward, int *terminal);
};

/// Equations of motion.
class Dynamics : public Configurable
{
  public:
    virtual ~Dynamics() { }
    virtual Dynamics *clone() const = 0;

    virtual void eom(const Vector &state, const Vector &action, Vector *xdd) const = 0;
};

class DynamicalModel : public Model
{
  public:
    TYPEINFO("model/dynamical")
    
  public:
    Dynamics *dynamics_;
    double tau_;
    size_t steps_;

  public:
    DynamicalModel() : dynamics_(NULL), tau_(0.05), steps_(5) { }
  
    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);
    
    // From Model
    virtual DynamicalModel *clone() const;
    virtual void step(const Vector &state, const Vector &action, Vector *next) const;
};

}

#endif /* GRL_ENVIRONMENT_H_ */
