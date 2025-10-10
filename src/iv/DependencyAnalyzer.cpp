/*********************                                                        */
/*! \file DependencyAnalyzer.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   <Your Name>
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Minimal implementation: store an owned copy of the base InputQuery.
 **/

#include "DependencyAnalyzer.h"

DependencyAnalyzer::DependencyAnalyzer( const InputQuery *baseIpq )
    : _baseIpq( baseIpq )
{
    // MVP: no heavy work here. Later we can derive signatures or
    // build auxiliary structures from baseIpq if needed.
}

DependencyAnalyzer::~DependencyAnalyzer() = default;

const InputQuery *DependencyAnalyzer::getBaseInputQuery() const
{
    return _baseIpq;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
