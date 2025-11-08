/*********************                                                        */
/*! \file DependencyState.cpp
** \verbatim
** Top contributors (to current version):
**   Raya Elsaleh
** This file is part of the Marabou project.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**/

#include "DependencyState.h"

DependencyState::DependencyState()
    : _depId( (unsigned)-1 )
{
}

DependencyState::DependencyState( DependencyId id, unsigned size )
    : _depId( id )
    , _current( size, ReLURuntimeState::Unstable )
{
}

DependencyState::DependencyId DependencyState::getDepId() const
{
    return _depId;
}

unsigned DependencyState::size() const
{
    return _current.size();
}

const Vector<ReLURuntimeState> &DependencyState::getCurrent() const
{
    return _current;
}

Vector<ReLURuntimeState> &DependencyState::getCurrent()
{
    return _current;
}
