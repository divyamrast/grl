/** \file agent.cpp
 * \brief ROS agent source file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2015-02-05
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

#include <grl/agents/ros.h>

using namespace grl;

REGISTER_CONFIGURABLE(ROSAgent)

void ROSAgent::request(ConfigurationRequest *config)
{
  config->push_back(CRP("node", "ROS node name", node_));
  config->push_back(CRP("args", "ROS command-line arguments", args_));
}

void ROSAgent::configure(Configuration &config)
{
  node_ = config["node"].str();
  args_ = config["args"].str();
  
  // Parse arguments
  int argc=1;
  char *argv[] = {"deployer"};
  
  ros::init(argc, argv, node_.c_str());
  
  nh_agent_ = new ros::NodeHandle("rl_agent");
  nh_env_ = new ros::NodeHandle("rl_env");
  
  action_sub_ = nh_agent_->subscribe("rl_action", 10, &ROSAgent::callbackAction, this);
  state_pub_ = nh_env_->advertise<mprl_msgs::StateReward>("rl_state_reward", 10, true);
  desc_pub_ = nh_env_->advertise<mprl_msgs::EnvDescription>("rl_env_description", 10, true);
  
  spinner_ = new ros::AsyncSpinner(0);
  spinner_->start();

  mprl_msgs::EnvDescription descmsg;

  Vector v;

  v = config["observation_min"];
  fromVector(v, descmsg.observation_min);
  v = config["observation_max"];
  fromVector(v, descmsg.observation_max);

  v = config["action_min"];
  action_dims_ = v.size();
  fromVector(v, descmsg.action_min);
  v = config["action_max"];
  fromVector(v, descmsg.action_max);

  descmsg.reward_min = config["reward_min"];
  descmsg.reward_max = config["reward_max"];

  descmsg.stochastic = config["stochastic"];
  descmsg.episodic = config["episodic"];
  descmsg.title = config["title"].str();

  INFO("Publishing environment description to ROS");

  desc_pub_.publish(descmsg);
  
  // Give agent some time to reinitialize
  usleep(100000);  
}

void ROSAgent::reconfigure(const Configuration &config)
{
}

ROSAgent *ROSAgent::clone() const
{
  return NULL;
}

void ROSAgent::start(const Vector &obs, Vector *action)
{
  if (running_)
  {
    mprl_msgs::StateReward statemsg;
    statemsg.reward = 0;
    statemsg.terminal = true;
    statemsg.absorbing = false;
    state_pub_.publish(statemsg);
  }	
	
  step(obs, 0, action);
  running_ = true;
}

void ROSAgent::step(const Vector &obs, double reward, Vector *action)
{
  Guard guard(mutex_);

  mprl_msgs::StateReward statemsg;
  statemsg.state.resize(obs.size());
  for (size_t ii=0; ii < obs.size(); ++ii)
    statemsg.state[ii] = obs[ii];
  statemsg.reward = reward;
  statemsg.terminal = false;
  statemsg.absorbing = false;

  state_pub_.publish(statemsg);
  
  new_action_.wait(mutex_);
  
  *action = action_;
}

void ROSAgent::end(double reward)
{
  mprl_msgs::StateReward statemsg;
  statemsg.reward = reward;
  statemsg.terminal = true;
  statemsg.absorbing = true;

  state_pub_.publish(statemsg);
  running_ = false;
}

void ROSAgent::callbackAction(const mprl_msgs::Action::ConstPtr &actionmsg)
{
  Guard guard(mutex_);

  toVector(actionmsg->action, action_);
  
  if (action_.size() != action_dims_)
    ERROR("Action received through ROS (size " << action_.size() << ") is not compatible with environment (size " << action_dims_ << ")");
  
  new_action_.signal();  
}
