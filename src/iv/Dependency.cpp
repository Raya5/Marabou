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

Dependency::Dependency()
    : _layer( 0 )
{
}

Dependency::Dependency( unsigned layer,
                        const std::vector<unsigned> &indices,
                        const std::vector<ReLUState> &states )
    : _layer( layer )
    , _indices( indices )
    , _states( states )
{
    canonicalize();
}

Dependency Dependency::Pair( unsigned layer,
                             unsigned a, unsigned b,
                             ReLUState aState, ReLUState bState )
{
    std::vector<unsigned> indices = { a, b };
    std::vector<ReLUState> states = { aState, bState };
    return Dependency( layer, indices, states );
}

Dependency Dependency::Triple( unsigned layer,
                               unsigned a, unsigned b, unsigned c,
                               ReLUState aState, ReLUState bState, ReLUState cState )
{
    std::vector<unsigned> indices = { a, b, c };
    std::vector<ReLUState> states = { aState, bState, cState };
    return Dependency( layer, indices, states );
}

unsigned Dependency::getLayer() const
{
    return _layer;
}

size_t Dependency::size() const
{
    return _indices.size();
}

const std::vector<unsigned> &Dependency::getIndices() const
{
    return _indices;
}

const std::vector<ReLUState> &Dependency::getStates() const
{
    return _states;
}

bool Dependency::isPair() const
{
    return _indices.size() == 2;
}

bool Dependency::isTriple() const
{
    return _indices.size() == 3;
}

bool Dependency::contains( unsigned neuron ) const
{
    return std::find( _indices.begin(), _indices.end(), neuron ) != _indices.end();
}

bool Dependency::operator==( const Dependency &other ) const
{
    return _layer == other._layer &&
           _indices == other._indices &&
           _states == other._states;
}

size_t Dependency::Hasher::operator()( const Dependency &d ) const
{
    size_t h = std::hash<unsigned>()( d._layer );
    for ( size_t i = 0; i < d._indices.size(); ++i )
    {
        h ^= std::hash<unsigned>()( d._indices[i] + 0x9e3779b9 + (h << 6) + (h >> 2) );
        h ^= std::hash<uint8_t>()( static_cast<uint8_t>( d._states[i] ) + (h << 5) );
    }
    return h;
}

void Dependency::canonicalize()
{
    if ( _indices.size() != _states.size() )
        return;

    std::vector<size_t> order( _indices.size() );
    for ( size_t i = 0; i < order.size(); ++i )
        order[i] = i;

    std::sort( order.begin(), order.end(),
               [&]( size_t a, size_t b ) { return _indices[a] < _indices[b]; } );

    std::vector<unsigned> sortedIndices;
    std::vector<ReLUState> sortedStates;
    sortedIndices.reserve( _indices.size() );
    sortedStates.reserve( _states.size() );

    for ( size_t k : order )
    {
        sortedIndices.push_back( _indices[k] );
        sortedStates.push_back( _states[k] );
    }

    _indices.swap( sortedIndices );
    _states.swap( sortedStates );
}
