/** \file leo_squatting_agent.cpp
 * \brief State-machine agent header file which performs squatting on Leo.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-02-04
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

#ifndef GRL_LEO_SQATTING_AGENT_H_
#define GRL_LEO_SQATTING_AGENT_H_

#include <grl/agent.h>
#include <grl/policy.h>
#include <grl/state_machine.h>

namespace grl
{

/// State machine agent.
class LeoSquattingAgent : public Agent
{
  public:
    TYPEINFO("mapping/policy/nmpc_sw", "Nonlinear model predictive control policy for the simplest walker using the MUSCOD library")

  protected:
    Agent *agent_standup_, *agent_learn_, *agent_;
    Trigger trigger_;
    double time_;
    
  public:
    LeoSquattingAgent() : agent_standup_(NULL), agent_learn_(NULL), agent_(NULL), time_(0.) { }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Agent
    virtual TransitionType start(const Vector &obs, Vector *action);
    virtual TransitionType step(double tau, const Vector &obs, double reward, Vector *action);
    virtual void end(double tau, const Vector &obs, double reward);
};

}

#endif /* GRL_LEO_SQATTING_AGENT_H_ */
