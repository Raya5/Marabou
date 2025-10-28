/*********************                                                        */
/*! \file Dependency.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Raya Elsaleh
 ** This file is part of the Marabou project.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** A generic representation of a same-layer dependency (forbidden ReLU state pattern).
 ** Each Dependency encodes a conflict between neuron activations, i.e., a set of ReLU
 ** states that cannot co-occur. It supports pairs, triples, and higher-order dependencies.
 **/

#ifndef __Dependency_h__
#define __Dependency_h__

#include <vector>
#include <cstddef>
#include <cstdint>

enum class ReLUState : uint8_t { Active, Inactive };

class Dependency
{
public:
    Dependency();
    Dependency( unsigned layer,
                const std::vector<unsigned> &indices,
                const std::vector<ReLUState> &states );

    // Convenience factories
    static Dependency Pair( unsigned layer,
                            unsigned a, unsigned b,
                            ReLUState aState, ReLUState bState );
    static Dependency Triple( unsigned layer,
                              unsigned a, unsigned b, unsigned c,
                              ReLUState aState, ReLUState bState, ReLUState cState );

    // Accessors
    unsigned getLayer() const;
    size_t size() const;
    const std::vector<unsigned> &getIndices() const;
    const std::vector<ReLUState> &getStates() const;

    bool isPair() const;
    bool isTriple() const;
    bool contains( unsigned neuron ) const;

    // Equality and hashing
    bool operator==( const Dependency &other ) const;

    struct Hasher
    {
        size_t operator()( const Dependency &d ) const;
    };

private:
    unsigned _layer;
    std::vector<unsigned> _indices;   // always sorted ascending
    std::vector<ReLUState> _states;   // aligned with _indices

    void canonicalize();
};

#endif // __Dependency_h__
