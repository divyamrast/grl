/** \file ros.h
 * \brief ROS agent header file.
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

#ifndef GRL_ROS_AGENT_H_
#define GRL_ROS_AGENT_H_

#include <ros/ros.h>
#include <mprl_msgs/Action.h>
#include <mprl_msgs/StateReward.h>
#include <mprl_msgs/EnvDescription.h>

#include <grl/mutex.h>
#include <grl/agent.h>

namespace grl
{

/// External agent interfaced via ROS.
class ROSAgent : public Agent
{
  public:
    TYPEINFO("agent/ros")

  protected:
    Mutex mutex_;
    Condition new_action_;
    bool running_;
    
    std::string node_, args_;
    
    size_t action_dims_;
    Vector action_;

    ros::NodeHandle *nh_agent_, *nh_env_;
    ros::Subscriber action_sub_;
    ros::Publisher state_pub_, desc_pub_;
    ros::AsyncSpinner *spinner_;
    
  public:
    ROSAgent() : running_(false), action_dims_(0), nh_agent_(NULL), nh_env_(NULL), spinner_(NULL) { }
    ~ROSAgent()
    {
      safe_delete(&nh_agent_);
      safe_delete(&nh_env_);
      safe_delete(&spinner_);
    }
  
    // From Configurable    
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Agent
    virtual ROSAgent *clone() const;
    virtual void start(const Vector &obs, Vector *action);
    virtual void step(const Vector &obs, double reward, Vector *action);
    virtual void end(double reward);
    
  protected:
    void callbackAction(const mprl_msgs::Action::ConstPtr &actionmsg);
};

}

#endif /* GRL_ROS_AGENT_H_ */
