/** \file rbf.h
 * \brief Triangular RBF projector header file.
 *
 * \author    Wouter Caarls <wouter@caarls.org>
 * \date      2016-11-29
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

#ifndef GRL_RBF_PROJECTOR_H_
#define GRL_RBF_PROJECTOR_H_

#include <grl/projector.h>

namespace grl
{

/// Standard discretization.
class RBFProjector : public Projector
{
  public:
    TYPEINFO("projector/rbf", "Projection on a grid of triangular radial basis functions")
    
  protected:
    Vector min_, max_, steps_, delta_;
    IndexVector stride_;
    
  public:
    RBFProjector() { }
  
    // From Configurable
    virtual void request(const std::string &role, ConfigurationRequest *config);
    virtual void configure(Configuration &config);
    virtual void reconfigure(const Configuration &config);

    // From Projector
    virtual ProjectionLifetime lifetime() const { return plIndefinite; }
    virtual ProjectionPtr project(const Vector &in) const;
};

}

#endif /* GRL_RBF_PROJECTOR_H_ */
