/*********************                                                        */
/*! \file Dependency.h
** \verbatim
** Top contributors (to current version):
**   Raya Elsaleh
** This file is part of the Marabou project.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**
** A generic representation of a dependency (forbidden ReLU state pattern).
**
** Each Dependency encodes a conflict between ReLU activations — a set of
** neuron states (Active / Inactive) that cannot co-occur.
**
** All variables refer to *Marabou variable IDs* (as returned by neuronToVariable()).
** The same structure supports same-layer and cross-layer dependencies.
**/

#ifndef __Dependency_h__
#define __Dependency_h__

#include <vector>
#include <cstddef>
#include <cstdint>
// #include <algorithm>

enum class ReLUState : uint8_t { Active, Inactive };

class Dependency
{
public:
    Dependency();
    Dependency( const std::vector<unsigned> &vars,
                const std::vector<ReLUState> &states );

    // Convenience factories
    static Dependency Pair( unsigned varA, unsigned varB,
                            ReLUState stateA, ReLUState stateB );
    static Dependency Triple( unsigned varA, unsigned varB, unsigned varC,
                              ReLUState stateA, ReLUState stateB, ReLUState stateC );

    // Accessors
    size_t size() const;
    const std::vector<unsigned> &getVars() const;
    const std::vector<ReLUState> &getStates() const;

    bool isPair() const;
    bool isTriple() const;
    bool contains( unsigned var ) const;

    // Equality and hashing
    bool operator==( const Dependency &other ) const;

    struct Hasher
    {
        size_t operator()( const Dependency &d ) const;
    };

private:
    std::vector<unsigned> _vars;    // always sorted ascending
    std::vector<ReLUState> _states; // aligned with _vars

    void canonicalize();
};

#endif // __Dependency_h__
