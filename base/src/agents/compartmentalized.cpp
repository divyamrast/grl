/** \file compartmentalized.cpp
 * \brief Compartmentalized sub-agent source file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-06-16
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

#include <grl/agents/compartmentalized.h>

using namespace grl;

REGISTER_CONFIGURABLE(CompartmentalizedSubAgent)

void CompartmentalizedSubAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("min", "vector.observation_min", "Minimum of compartment bounding box", min_));
  config->push_back(CRP("max", "vector.observation_max", "Maximum of compartment bounding box", max_));
  config->push_back(CRP("agent", "agent", "Sub agent", agent_));
}

void CompartmentalizedSubAgent::configure(Configuration &config)
{
  agent_ = (Agent*)config["agent"].ptr();

  min_ = config["min"].v();
  max_ = config["max"].v();
  
  if (min_.size() != max_.size())
    throw bad_param("agent/sub/compartmentalized:{min,max}");
}

void CompartmentalizedSubAgent::reconfigure(const Configuration &config)
{
}

void CompartmentalizedSubAgent::start(const Observation &obs, Action *action)
{
  agent_->start(obs, action);
}

void CompartmentalizedSubAgent::step(double tau, const Observation &obs, double reward, Action *action)
{
  agent_->step(tau, obs, reward, action);
}

void CompartmentalizedSubAgent::end(double tau, const Observation &obs, double reward)
{
  agent_->end(tau, obs, reward);
}

double CompartmentalizedSubAgent::confidence(const Observation &obs) const
{
  if (!min_.size())
    return 0.999;

  if (obs.size() != min_.size())
    throw bad_param("agent/sub/compartmentalized:{min,max}");
    
  bool in=true;
  for (size_t ii=0; ii != obs.size(); ++ii)
    if (obs[ii] < min_[ii] || obs[ii] > max_[ii])
      in = false;
      
  return 1.*in;
}
