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
#include "Debug.h"                  // for ASSERT

DependencyState::DependencyState()
    : _depId( (unsigned)-1 )
    , _hasImplication(false)
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

void DependencyState::setActive( unsigned i )
{
    ASSERT( i < _current.size() );
    ASSERT( _current[i] == ReLURuntimeState::Unstable );
    _current[i] = ReLURuntimeState::Active;
}

void DependencyState::setInactive( unsigned i )
{
    ASSERT( i < _current.size() );
    ASSERT( _current[i] == ReLURuntimeState::Unstable );
    _current[i] = ReLURuntimeState::Inactive;
}

static inline ReLUState negatePhase( ReLUState s )
{
    return ( s == ReLUState::Active ) ? ReLUState::Inactive : ReLUState::Active;
}

bool DependencyState::checkImplication( const Dependency &dep) 
{
    // 2) One-away check
    const auto &vars   = dep.getVars();
    const auto &phases = dep.getStates();
    ASSERT( vars.size() == phases.size() );
    ASSERT( vars.size() == _current.size() );

    unsigned matched = 0, contradicted = 0, unset = 0;
    int lastUnsetIdx = -1;

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const auto rt = _current[i];
        if ( rt == ReLURuntimeState::Unstable ) { ++unset; lastUnsetIdx = i; continue; }

        const ReLUState rtAsPhase =
            ( rt == ReLURuntimeState::Active ) ? ReLUState::Active : ReLUState::Inactive;

        if ( rtAsPhase == phases[i] ) ++matched;
        else                          ++contradicted;
    }

    ASSERT( matched + contradicted + unset == dep.size() );

    if ( contradicted == 0 && unset == 1 )
    {
        _hasImplication = true;
        _impVar   = vars[lastUnsetIdx];
        _impPhase = negatePhase( phases[lastUnsetIdx] ); // opposite to nogood polarity
        return true;
    }

    return false;
}

bool DependencyState::hasImplication() const { return _hasImplication; }

void DependencyState::getImplication( unsigned &var, ReLUState &phase ) const
{
    ASSERT( _hasImplication );
    var = _impVar; phase = _impPhase;
}
