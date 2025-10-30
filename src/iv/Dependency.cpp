/*********************                                                        */
/*! \file Dependency.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Raya Elsaleh
 ** This file is part of the Marabou project.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "Dependency.h"
#include <functional>
// #include <algorithm>

Dependency::Dependency()
{
}

Dependency::Dependency( const std::vector<unsigned> &vars,
                        const std::vector<ReLUState> &states )
    : _vars( vars )
    , _states( states )
{
    canonicalize();
}

Dependency Dependency::Pair( unsigned varA, unsigned varB,
                             ReLUState stateA, ReLUState stateB )
{
    std::vector<unsigned> vars   = { varA, varB };
    std::vector<ReLUState> states = { stateA, stateB };
    return Dependency( vars, states );
}

Dependency Dependency::Triple( unsigned varA, unsigned varB, unsigned varC,
                               ReLUState stateA, ReLUState stateB, ReLUState stateC )
{
    std::vector<unsigned> vars   = { varA, varB, varC };
    std::vector<ReLUState> states = { stateA, stateB, stateC };
    return Dependency( vars, states );
}

size_t Dependency::size() const
{
    return _vars.size();
}

const std::vector<unsigned> &Dependency::getVars() const
{
    return _vars;
}

const std::vector<ReLUState> &Dependency::getStates() const
{
    return _states;
}

bool Dependency::isPair() const
{
    return _vars.size() == 2;
}

bool Dependency::isTriple() const
{
    return _vars.size() == 3;
}

bool Dependency::contains( unsigned var ) const
{
    return std::find( _vars.begin(), _vars.end(), var ) != _vars.end();
}

bool Dependency::operator==( const Dependency &other ) const
{
    return _vars == other._vars && _states == other._states;
}

size_t Dependency::Hasher::operator()( const Dependency &d ) const
{
    size_t h = 0;
    for ( size_t i = 0; i < d._vars.size(); ++i )
    {
        // Combine variable ID and state into hash
        h ^= std::hash<unsigned>()( d._vars[i] )
             + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<uint8_t>()( static_cast<uint8_t>( d._states[i] ) )
             + 0x9e3779b97f4a7c15ULL + (h << 5) + (h >> 2);
    }
    return h;
}

void Dependency::canonicalize()
{
    if ( _vars.size() != _states.size() )
        return;

    std::vector<size_t> order( _vars.size() );
    for ( size_t i = 0; i < order.size(); ++i )
        order[i] = i;

    std::sort( order.begin(), order.end(),
               [&]( size_t a, size_t b ) { return _vars[a] < _vars[b]; } );

    std::vector<unsigned> sortedVars;
    std::vector<ReLUState> sortedStates;
    sortedVars.reserve( _vars.size() );
    sortedStates.reserve( _states.size() );

    for ( size_t k : order )
    {
        sortedVars.push_back( _vars[k] );
        sortedStates.push_back( _states[k] );
    }

    _vars.swap( sortedVars );
    _states.swap( sortedStates );
}
