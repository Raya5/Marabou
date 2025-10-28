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
#include "Tightening.h"              // for Tightening
#include "FloatUtils.h"              // for gt/lt comparisons
#include "Layer.h"

DependencyAnalyzer::DependencyAnalyzer( const InputQuery *baseIpq )
    : _baseIpq( baseIpq )
    , _preprocessedQuery( nullptr )
    , _networkLevelReasoner( nullptr )
{
    buildFromBase();
    std::printf("[DA] initial _baseIpq: vars=%u, eqs=%u\n",
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
    unsigned numTightened = runBoundTightening();
    printf("[DA] first DeepPoly tightening: %u tightenings\n", numTightened);
    computeSameLayerDependencies(1);
    computeSameLayerDependencies(3);
    // (no tableau hookup, no dumps)
    // debugdiff()
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

unsigned DependencyAnalyzer::runBoundTightening()
{
    if ( !_preprocessedQuery || !_networkLevelReasoner )
    {
        printf("runBoundTightening called before buildFromBase()");
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
            "runBoundTightening called before buildFromBase()" );
    }

    // 1) Run DeepPoly over the current bounds held by the preprocessed query / NLR.
    _networkLevelReasoner->deepPolyPropagation();

    // 2) Collect proposed tightenings.
    List<Tightening> tightenings;
    _networkLevelReasoner->getConstraintTightenings( tightenings );

    // 3) Apply them back to the preprocessed query (like Engine does).
    unsigned numTightened = 0;
    for ( const auto &t : tightenings )
    {
        const unsigned v = t._variable;
        const double   x = t._value;
        if ( t._type == Tightening::LB )
        {
            if ( FloatUtils::gt( x, _preprocessedQuery->getLowerBound( v ) ) )
            {
                _preprocessedQuery->setLowerBound( v, x );
                ++numTightened;
            }
        }
        else /* UB */
        {
            if ( FloatUtils::lt( x, _preprocessedQuery->getUpperBound( v ) ) )
            {
                _preprocessedQuery->setUpperBound( v, x );
                ++numTightened;
            }
        }
    }
    return numTightened;
}

void DependencyAnalyzer::collectUnstableNeurons( unsigned layerIndex,
                                                 std::vector<unsigned> &unstableNeurons ) const
{
    unstableNeurons.clear();

    if ( !_networkLevelReasoner ) {
        printf("[DA] collectUnstableNeurons: NLR not set\n");
        return;
    }

    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( layerIndex );
    if ( !weightedSumLayer ) {
        printf("[DA] collectUnstableNeurons: layer %u not found in NLR\n", layerIndex);
        return;
    }

    const unsigned numNeurons = weightedSumLayer->getSize();

    // A neuron is considered "unstable" if its pre-activation interval straddles 0.
    for ( unsigned neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex )
    {
        double lowerPreActivation = weightedSumLayer->getLb( neuronIndex );
        double upperPreActivation = weightedSumLayer->getUb( neuronIndex );
        //################### FOR DEBUGING ########################
        unsigned v = weightedSumLayer->neuronToVariable( neuronIndex );

        double nlrLb = weightedSumLayer->getLb( neuronIndex );
        double pqLb  = _preprocessedQuery->getLowerBound( v );

        double nlrUb = weightedSumLayer->getUb( neuronIndex );
        double pqUb  = _preprocessedQuery->getUpperBound( v );

        if ( !FloatUtils::areEqual( nlrLb, pqLb, 1e-9 ) ||
            !FloatUtils::areEqual( nlrUb, pqUb, 1e-9 ) )
        {
            printf("[DA][warn] bound mismatch at layer %u neuron %u "
                "(var %u): NLR [%.6g, %.6g] vs PQ [%.6g, %.6g]\n",
                layerIndex, neuronIndex, v, nlrLb, nlrUb, pqLb, pqUb);
        }

        //################### END DEBUGING ########################

        if ( lowerPreActivation < 0.0 && upperPreActivation > 0.0 )
            unstableNeurons.push_back( neuronIndex );
    }

    printf("[DA] Layer %u: found %zu unstable neurons\n", layerIndex, unstableNeurons.size());
}


unsigned DependencyAnalyzer::computeSameLayerDependencies( unsigned weightedSumLayerIndex )
{
    if ( !_networkLevelReasoner ) {
        printf("[DA] computeSameLayerDependencies: NLR not set\n");
        return 0;
    }

    // ensure NLR bounds are current (you already run DeepPoly elsewhere; this is cheap)
    _networkLevelReasoner->obtainCurrentBounds(*_preprocessedQuery); 

    // sanity: layer must be WEIGHTED_SUM (pre-activation layer)
    const auto *weightedSumLayer = _networkLevelReasoner->getLayer( weightedSumLayerIndex );
    if ( !weightedSumLayer || weightedSumLayer->getLayerType() != NLR::Layer::WEIGHTED_SUM ) {
        printf("[DA] layer %u is not WEIGHTED_SUM\n", weightedSumLayerIndex);
        return 0;
    }
    
    // build unstable set from pre-activation bounds of this layer
    std::vector<unsigned> unstable;
    collectUnstableNeurons( weightedSumLayerIndex, unstable );
    if ( unstable.size() < 2 ) return 0;

    // auto &bucket = _pairsByLayer[ weightedSumLayerIndex ];
    unsigned added = 0;
    
    // enumerate unordered pairs
    for ( size_t i = 0; i + 1 < unstable.size(); ++i ) {
        unsigned q = unstable[i];
        for ( size_t j = i + 1; j < unstable.size(); ++j ) {
            unsigned r = unstable[j];
            ASSERT(q < r);
            if (detectAndRecordPairConflict(weightedSumLayerIndex, q, r))
                ++added;
        }
    }
    printf("[DA] layer %u: scanned %zu pairs, added %u conflicts\n",
           weightedSumLayerIndex, ( unstable.size() * ( unstable.size() - 1 ) ) / 2, added );

    printf("*******************  ACHIEVED  ******************* \n");
    return added;
}


bool DependencyAnalyzer::detectAndRecordPairConflict(unsigned layerIndex,
                                 unsigned q, unsigned r)
{
    Dependency d;
    if (!analyzePairConflict(layerIndex, q, r, d))
        return false;
    recordConflict(std::move(d));
    return true;
}

/*
  Placeholder for analyzing a pair (q,r) of neurons in the same layer.
  Later this will compute conditional intervals and determine whether
  a dependency (forbidden combination) exists.
*/
bool DependencyAnalyzer::analyzePairConflict( unsigned layer,
                                              unsigned q, unsigned r,
                                              Dependency &outDependency )
{
    (void)layer;
    (void)q;
    (void)r;
    (void)outDependency;

    // TODO: implement the mathematical analysis for size-2 dependencies.
    printf("[DA] analyzePairConflict(%u, %u, %u) -- not yet implemented\n", layer, q, r);
    return false; // No dependency detected (placeholder)
}

/*
  Placeholder for recording a dependency in the internal storage.
  Later this will insert into the per-layer dependency set/map,
  handling canonicalization and duplicate detection.
*/
bool DependencyAnalyzer::recordConflict( Dependency d )
{
    (void)d;

    // TODO: implement dependency insertion and deduplication.
    printf("[DA] recordConflict() -- not yet implemented\n");
    return false; // Return true if newly inserted (placeholder)
}






/**************** For Debugging ********************/
// DependencyAnalyzer.cpp (using _preprocessedQuery)
BoundsSnapshot DependencyAnalyzer::snapshotBounds(const std::vector<unsigned> &vars) {
    BoundsSnapshot s;
    if (!vars.empty()) {
        for (auto v : vars)
            s.byVar[v] = { _preprocessedQuery->getLowerBound(v),
                           _preprocessedQuery->getUpperBound(v) };
    } else {
        const auto n = _preprocessedQuery->getNumberOfVariables();
        for (unsigned v = 0; v < n; ++v)
            s.byVar[v] = { _preprocessedQuery->getLowerBound(v),
                           _preprocessedQuery->getUpperBound(v) };
    }
    return s;
}

std::vector<std::tuple<unsigned,double,double,double,double>>
DependencyAnalyzer::diffBounds(const BoundsSnapshot &a, const BoundsSnapshot &b, double eps) {
    std::vector<std::tuple<unsigned,double,double,double,double>> out;
    for (const auto &kv : b.byVar) {
        auto v = kv.first;
        auto [lb2, ub2] = kv.second;
        auto it = a.byVar.find(v);
        if (it == a.byVar.end()) continue;
        auto [lb1, ub1] = it->second;
        bool chgLb = std::fabs(lb2 - lb1) > eps;
        bool chgUb = std::fabs(ub2 - ub1) > eps;
        if (chgLb || chgUb) out.emplace_back(v, lb1, ub1, lb2, ub2);
    }
    return out;
}

void DependencyAnalyzer::printBoundsDiff(const std::vector<std::tuple<unsigned,double,double,double,double>> &d,
                                         unsigned maxItems) {
    unsigned shown = 0;
    for (const auto &t : d) {
        if (shown++ >= maxItems) { printf("... truncated ...\n"); break; }
        unsigned v; double lb1, ub1, lb2, ub2;
        std::tie(v, lb1, ub1, lb2, ub2) = t;
        printf("[DA] v=%u: LB %.6g -> %.6g | UB %.6g -> %.6g\n", v, lb1, lb2, ub1, ub2);
    }
    if (d.empty()) printf("[DA] no bound changes\n");
}

void DependencyAnalyzer::debugdiff()
{
auto snap1 = snapshotBounds();
runBoundTightening();
auto snap2 = snapshotBounds();
auto diff  = diffBounds(snap1, snap2, 1e-9);
printBoundsDiff(diff, 40);
}
/**************** End For Debugging ********************/



//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
