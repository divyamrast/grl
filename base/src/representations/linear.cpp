/** \file linear.cpp
 * \brief Linear representation source file.
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

#include <grl/representations/linear.h>

using namespace grl;

REGISTER_CONFIGURABLE(LinearRepresentation)

void LinearRepresentation::request(ConfigurationRequest *config)
{
  config->push_back(CRP("memory", "Feature vector size", (int)memory_, CRP::System));
  config->push_back(CRP("outputs", "Number of outputs", (int)outputs_, CRP::System));

  config->push_back(CRP("init_min", "Lower initial value limit", init_min_));
  config->push_back(CRP("init_max", "Upper initial value limit", init_max_));
  
  config->push_back(CRP("output_min", "Lower output limit", output_min_, CRP::System));
  config->push_back(CRP("output_max", "Upper output limit", output_max_, CRP::System));
}

void LinearRepresentation::configure(Configuration &config)
{
  memory_ = config["memory"];
  outputs_ = config["outputs"];
  
  init_min_ = config["init_min"];
  if (init_min_.size() && init_min_.size() < outputs_)
    init_min_.resize(outputs_, init_min_[0]);
  if (init_min_.size() != outputs_)
    throw bad_param("representation/parameterized/linear:init_min");
  
  init_max_ = config["init_max"];
  if (init_max_.size() && init_max_.size() < outputs_)
    init_max_.resize(outputs_, init_max_[0]);
  if (init_max_.size() != outputs_)
    throw bad_param("representation/parameterized/linear:max");

  params_.resize(memory_ * outputs_);
  
  output_min_ = config["output_min"];
  if (output_min_.empty())
    output_min_.resize(outputs_, -DBL_MAX);
  if (output_min_.size() != outputs_)
    throw bad_param("representation/parameterized/linear:output_min");
    
  output_max_ = config["output_max"];
  if (output_max_.empty())
    output_max_.resize(outputs_, DBL_MAX);
  if (output_max_.size() != outputs_)
    throw bad_param("representation/parameterized/linear:output_max");
  
  // Initialize memory
  reset();
}

void LinearRepresentation::reconfigure(const Configuration &config)
{
  if (config.has("action") && config["action"].str() == "reset")
  {
    DEBUG("Initializing " << memory_ << " values between " << init_min_ << " and " << init_max_);
  
    params_.resize(memory_ * outputs_);

    // Initialize memory
    Rand *rand = RandGen::instance();
    for (size_t ii=0; ii < memory_; ++ii)
      for (size_t jj=0; jj < outputs_; ++jj)  
        params_[ii*outputs_+jj] = rand->getUniform(init_min_[jj], init_max_[jj]);
  }
}

LinearRepresentation *LinearRepresentation::clone() const
{
  return new LinearRepresentation(*this);
}

double LinearRepresentation::read(const ProjectionPtr &projection, Vector *result) const
{
  Projection &p = *projection;
  
  IndexProjection *ip = dynamic_cast<IndexProjection*>(&p);
  if (ip)
  {
    result->clear();
    result->resize(outputs_, 0);
    
    for (size_t ii=0; ii < ip->indices.size(); ++ii)
      for (size_t jj=0; jj < outputs_; ++jj)
        (*result)[jj] += params_[ip->indices[ii]*outputs_+jj];
    for (size_t jj=0; jj < outputs_; ++jj)
      (*result)[jj] /= ip->indices.size();
  }
  else
  {
    VectorProjection *vp = dynamic_cast<VectorProjection*>(&p);
    if (vp)
    {
      if (vp->vector.size() != memory_)
        throw bad_param("representation/parameterized/linear:memory (or matching projector)");
    
      result->clear();
      result->resize(outputs_, 0);
      
      for (size_t ii=0; ii < vp->vector.size(); ++ii)
        for (size_t jj=0; jj < outputs_; ++jj)
          (*result)[jj] += params_[ii*outputs_+jj]*vp->vector[ii];
    }
    else
      throw Exception("representation/parameterized/linear requires a projector returning IndexProjection or VectorProjection");
  }

  for (size_t ii=0; ii < outputs_; ++ii)
    (*result)[ii] = fmin(fmax((*result)[ii], output_min_[ii]), output_max_[ii]);
  
  return (*result)[0];
}

void LinearRepresentation::write(const ProjectionPtr projection, const Vector &target, const Vector &alpha)
{
  if (target.size() != alpha.size())
    throw Exception("Learning rate vector does not match target vector");

  // TODO: Store read values and update those (for thread safety)
  Vector value;
  read(projection, &value);
  Vector delta = alpha*(target-value);
  
  update(projection, delta);
}

void LinearRepresentation::update(const ProjectionPtr projection, const Vector &delta)
{
  Projection &p = *projection;
  
  IndexProjection *ip = dynamic_cast<IndexProjection*>(&p);
  if (ip)
  {
    for (size_t ii=0; ii != ip->indices.size(); ++ii)
      for (size_t jj=0; jj < outputs_; ++jj)
        if (ip->indices[ii] != IndexProjection::invalid_index())
          params_[ip->indices[ii]*outputs_+jj] = fmin(fmax(params_[ip->indices[ii]*outputs_+jj] + delta[jj], output_min_[jj]), output_max_[jj]);
  }
  else
  {
    VectorProjection *vp = dynamic_cast<VectorProjection*>(&p);
    if (vp)
    {
      if (vp->vector.size() != memory_)
        throw bad_param("representation/parameterized/linear:memory (or matching projector)");

      for (size_t ii=0; ii != vp->vector.size(); ++ii)
        for (size_t jj=0; jj < outputs_; ++jj)
          params_[ii*outputs_+jj] = fmin(fmax(params_[ii*outputs_+jj] + vp->vector[ii]*delta[jj], output_min_[jj]), output_max_[jj]);
    }
    else
      throw Exception("representation/parameterized/linear requires a projector returning IndexProjection or VectorProjection");
  }
}
