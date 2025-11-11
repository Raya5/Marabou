/*********************                                                        */
/*! \file DependencyState.h
** \verbatim
** Top contributors (to current version):
**   Raya Elsaleh
** This file is part of the Marabou project.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**
** A lightweight runtime state tracker for a single dependency.
**
** Each DependencyState maintains, for one dependency (identified by depId),
** the current ReLU runtime state (Active / Inactive / Unstable) of each literal.
**
** The `_current` vector is always index-aligned with the Dependency’s literal list.
**
** Used by DependencyAnalyzer for incremental reasoning and propagation.
**/

#ifndef __DependencyState_h__
#define __DependencyState_h__

#include "Vector.h"
#include "Dependency.h"
// #include <cstdint>

/*
  Possible runtime states for a ReLU literal within a dependency.
*/
enum class ReLURuntimeState : uint8_t { Unstable, Active, Inactive };

/*
  DependencyState:
  Tracks the runtime activation status for all literals in a single dependency.
  Each element in `_current` corresponds to one literal in the dependency.
*/
class DependencyState
{
public:
    typedef unsigned DependencyId;

    /*
      Default constructor.
      Needed for STL container initialization.
    */
    DependencyState();

    /*
      Construct a DependencyState for a given dependency id,
      initializing all literal states to Unstable.
    */
    DependencyState( DependencyId id, unsigned size );

    /*
      Return the owning dependency id.
    */
    DependencyId getDepId() const;

    /*
      Number of literals (same as dependency size).
    */
    unsigned size() const;

    /*
      Accessors for the literal runtime state vector.
    */
    const Vector<ReLURuntimeState> &getCurrent() const;
    Vector<ReLURuntimeState> &getCurrent();

    /*
      Check whether this dependency is now "one-away" given the current runtime
      phases in `current`. If so, return true and output the single implied
      (var, phase) that must be forced to avoid violating the nogood.

      Returns:
        - true  => exactly one literal is still Unstable, all others are fixed,
                  none contradicts the nogood; (outVar, outPhase) is set to
                  the forced opposite phase of that last literal.
        - false => otherwise (no implication).
    */
    bool checkImplication( const Dependency &dep );
    bool hasImplication() const;
    void getImplication( unsigned &var, ReLUState &phase ) const;

private:
    DependencyId _depId;                 // Index of the owning dependency
    Vector<ReLURuntimeState> _current;   // index-aligned literal states

    bool _hasImplication;
    unsigned _impVar;
    ReLUState _impPhase;

};

#endif // __DependencyState_h__
