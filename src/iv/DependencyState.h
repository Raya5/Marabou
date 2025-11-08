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

private:
    DependencyId _depId;                 // Index of the owning dependency
    Vector<ReLURuntimeState> _current;   // index-aligned literal states
};

#endif // __DependencyState_h__
