/** \file mapping.h
 * \brief Q-policy mapping definition.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2016-06-01
 *
 * \copyright \verbatim
 * Copyright (c) 2016, Wouter Caarls
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

#ifndef GRL_Q_POLICY_MAPPING_H_
#define GRL_Q_POLICY_MAPPING_H_

#include <grl/mapping.h>
#include <grl/policies/q.h>

namespace grl
{

class QPolicyMapping : public Mapping
{
  public:
    TYPEINFO("mapping/q_policy", "Mapping that returns the value of a q-policy")

  protected:
    QPolicy *policy_;
  
  public:
    QPolicyMapping() : policy_(NULL)
    {
    }
  
    // From Configurable
    virtual void request(ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Mapping
    virtual double read(const Vector &in, Vector *result) const;
};

}

#endif /* GRL_Q_POLICY_MAPPING_H_ */
