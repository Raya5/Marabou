/*********************                                                        */
/*! \file DependencyAnalyzer.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Raya E. 
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** Minimal implementation: store an owned copy of the base InputQuery.
 **/

#include "DependencyAnalyzer.h"
#include "Preprocessor.h"
#include "GlobalConfiguration.h"
// #include "NetworkLevelReasoner.h"

DependencyAnalyzer::DependencyAnalyzer( const InputQuery *baseIpq )
    : _baseIpq( baseIpq )
    , _preprocessedQuery( nullptr )
    , _networkLevelReasoner( nullptr )
{
    buildFromBase();
    std::printf("[DA] initial _baseIpq: vars=%u, eqs=%u",
        _baseIpq->getNumberOfVariables(),
        _baseIpq->getNumberOfEquations());
}

void DependencyAnalyzer::buildFromBase()
{
    if ( !_baseIpq )
    {
        printf("DependencyAnalyzer::buildFromBase called with null baseIpq." );
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
            "DependencyAnalyzer::buildFromBase called with null baseIpq." );
            
    }

    Preprocessor preprocessor;

    // preprocess returns std::unique_ptr<Query>; assign/move it directly
    _preprocessedQuery = preprocessor.preprocess(
        *_baseIpq, GlobalConfiguration::PREPROCESSOR_ELIMINATE_VARIABLES );


    // Cache the NLR owned by the preprocessed query
    _networkLevelReasoner =
        _preprocessedQuery ? _preprocessedQuery->getNetworkLevelReasoner() : nullptr;

    if ( !_networkLevelReasoner )
    {
        printf("Preprocessing failed: NetworkLevelReasoner is null." );
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Preprocessing failed: NetworkLevelReasoner is null." );
    }
    _networkLevelReasoner->computeSuccessorLayers();
    // (no tableau hookup, no dumps)
}

DependencyAnalyzer::~DependencyAnalyzer() = default;

const InputQuery *DependencyAnalyzer::getBaseInputQuery() const
{
    return _baseIpq;
}


void DependencyAnalyzer::printSummary() const
{
    // Lightweight, safe diagnostic
    std::printf(
        "[DependencyAnalyzer] baseIpq=%p, numVars=%s, preprocessed=%s, nlr=%s\n",
        (const void *)_baseIpq,
        _baseIpq ? "known" : "unknown",
        _preprocessedQuery ? "yes" : "no",
        _networkLevelReasoner ? "yes" : "no"
    );
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
