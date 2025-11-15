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
{
}

DependencyState::DependencyState( DependencyId depId,
                                  unsigned numLiterals,
                                  CVC4::context::Context &ctx )
    : _depId( depId )
{
    _current.clear();
    for ( unsigned i = 0; i < numLiterals; ++i )
    {
        _current.append(
            new (true) CVC4::context::CDO<ReLURuntimeState>( &ctx ));
            // new CVC4::context::CDO<ReLURuntimeState>( &ctx,
            //                                           ReLURuntimeState::Unstable ) );
        *_current[_current.size() -1 ] = ReLURuntimeState::Unstable;
    }
}

DependencyState::~DependencyState() = default;

DependencyState::DependencyId DependencyState::getDepId() const
{
    return _depId;
}

unsigned DependencyState::size() const
{
    return _current.size();
}

ReLURuntimeState DependencyState::getLiteralState( unsigned i ) const
{
    ASSERT( i < _current.size() );
    return *_current[i];                     // CDO<T> → T via operator T()
}

void DependencyState::setActive( unsigned i )
{
    ASSERT( i < _current.size() );
    ASSERT( *_current[i] == ReLURuntimeState::Unstable );
    *_current[i] = ReLURuntimeState::Active;
}

void DependencyState::setInactive( unsigned i )
{
    ASSERT( i < _current.size() );
    ASSERT( *_current[i] == ReLURuntimeState::Unstable );
    *_current[i] = ReLURuntimeState::Inactive;
}

static inline ReLUState negatePhase( ReLUState s )
{
    return ( s == ReLUState::Active ) ? ReLUState::Inactive : ReLUState::Active;
}

bool DependencyState::checkImplication( const Dependency &dep,
                                        unsigned &outVar,
                                        ReLUState &outPhase ) const
{
    const auto &vars   = dep.getVars();
    const auto &phases = dep.getStates();
    ASSERT( vars.size() == phases.size() );
    ASSERT( vars.size() == _current.size() );

    unsigned matched = 0, contradicted = 0, unset = 0;
    int lastUnsetIdx = -1;

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const ReLURuntimeState rt = getLiteralState( i );

        if ( rt == ReLURuntimeState::Unstable )
        {
            ++unset;
            lastUnsetIdx = static_cast<int>( i );
            continue;
        }

        const ReLUState rtAsPhase =
            ( rt == ReLURuntimeState::Active ) ? ReLUState::Active
                                               : ReLUState::Inactive;

        if ( rtAsPhase == phases[i] )
            ++matched;
        else
            ++contradicted;
    }

    ASSERT( matched + contradicted + unset == dep.size() );

    // One-away nogood: exactly one literal unset, none contradicted
    if ( contradicted == 0 && unset == 1 )
    {
        ASSERT( lastUnsetIdx >= 0 );
        outVar   = vars[ lastUnsetIdx ];
        outPhase = negatePhase( phases[ lastUnsetIdx ] ); // opposite to nogood polarity
        return true;
    }

    return false;
}

