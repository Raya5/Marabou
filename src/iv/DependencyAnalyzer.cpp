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
bool DependencyAnalyzer::analyzePairConflict( unsigned layerIndex,
                                              unsigned q, unsigned r,
                                              Dependency &outDependency )
{
    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( layerIndex );
    ASSERT( weightedSumLayer );
    ASSERT( weightedSumLayer->getLayerType() == NLR::Layer::WEIGHTED_SUM );

    const unsigned layerSize = weightedSumLayer->getSize();
    ASSERT( q < layerSize && r < layerSize );
    ASSERT( q < r );
    (void) layerSize;

    const auto &sources = weightedSumLayer->getSourceLayers();
    ASSERT( sources.size() == 1 );
    const unsigned prevLayerIndex = sources.begin()->first;
    const NLR::Layer *prevLayer = _networkLevelReasoner->getLayer( prevLayerIndex );
    ASSERT( prevLayer );
    const unsigned prevSize = prevLayer->getSize();

    // === Collect weight rows and biases ===
    Vector<double> w_q( prevSize ), w_r( prevSize );
    for ( unsigned j = 0; j < prevSize; ++j )
    {
        w_q[j] = weightedSumLayer->getWeight( prevLayerIndex, j, q );
        w_r[j] = weightedSumLayer->getWeight( prevLayerIndex, j, r );
    }

    const double b_q = weightedSumLayer->getBias( q );
    const double b_r = weightedSumLayer->getBias( r );

    Vector<double> lowerPrev, upperPrev;
    _getLayerBounds( prevLayer, lowerPrev, upperPrev );

    // === Compute conditional bounds ===
    double l_q_r0, u_q_r0, l_r_q0, u_r_q0;
    _sliceMinMax_givenOtherZero( w_q, b_q, w_r, b_r, lowerPrev, upperPrev, l_q_r0, u_q_r0 );
    _sliceMinMax_givenOtherZero( w_r, b_r, w_q, b_q, lowerPrev, upperPrev, l_r_q0, u_r_q0 );

    // === Debug info ===
    unsigned countTrue = 0;
    if ( FloatUtils::gt( l_q_r0, 0.0 ) ) ++countTrue;
    if ( FloatUtils::lt( u_q_r0, 0.0 ) ) ++countTrue;
    if ( FloatUtils::gt( l_r_q0, 0.0 ) ) ++countTrue;
    if ( FloatUtils::lt( u_r_q0, 0.0 ) ) ++countTrue;

    if ( countTrue > 0 )
    {
        unsigned varQ = weightedSumLayer->neuronToVariable( q );
        unsigned varR = weightedSumLayer->neuronToVariable( r );

        printf("[DA][pair %u,%u] (vars %u,%u) >0(l_q_r0)=%d  <0(u_q_r0)=%d  "
               ">0(l_r_q0)=%d  <0(u_r_q0)=%d  totalTrue=%u\n",
               q, r, varQ, varR,
               FloatUtils::gt( l_q_r0, 0.0 ),
               FloatUtils::lt( u_q_r0, 0.0 ),
               FloatUtils::gt( l_r_q0, 0.0 ),
               FloatUtils::lt( u_r_q0, 0.0 ),
               countTrue );
    }
    // === End of Debug info ===

    // === Boolean classification ===
    const bool q_forced_active   = FloatUtils::gt( l_q_r0, 0.0 );
    const bool q_forced_inactive = FloatUtils::lt( u_q_r0, 0.0 );
    const bool r_forced_active   = FloatUtils::gt( l_r_q0, 0.0 );
    const bool r_forced_inactive = FloatUtils::lt( u_r_q0, 0.0 );

    unsigned varQ = weightedSumLayer->neuronToVariable( q );
    unsigned varR = weightedSumLayer->neuronToVariable( r );

    // === Create forbidden combination (dependency) ===
    if ( q_forced_inactive && r_forced_inactive )
    {
        // u_q|r0 < 0 and u_r|q0 < 0 ⇒ forbid (q=Active, r=Active)
        outDependency = Dependency::Pair( varQ, varR,
                                          ReLUState::Active, ReLUState::Active );
    }
    else if ( q_forced_active && r_forced_active )
    {
        // l_q|r0 > 0 and l_r|q0 > 0 ⇒ forbid (q=Inactive, r=Inactive)
        outDependency = Dependency::Pair( varQ, varR,
                                          ReLUState::Inactive, ReLUState::Inactive );
    }
    else if ( q_forced_inactive && r_forced_active )
    {
        // u_q|r0 < 0 and l_r|q0 > 0 ⇒ forbid (q=Active, r=Inactive)
        outDependency = Dependency::Pair( varQ, varR,
                                          ReLUState::Active, ReLUState::Inactive );
    }
    else if ( q_forced_active && r_forced_inactive )
    {
        // l_q|r0 > 0 and u_r|q0 < 0 ⇒ forbid (q=Inactive, r=Active)
        outDependency = Dependency::Pair( varQ, varR,
                                          ReLUState::Inactive, ReLUState::Active );
    }
    else
    {
        return false; // no dependency found
    }

    return true; // dependency found and written to outDependency
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

void DependencyAnalyzer::_getLayerBounds( const NLR::Layer *layer,
                                          Vector<double> &lowerBounds,
                                          Vector<double> &upperBounds ) const
{
    Vector<double> L, U;
    const unsigned n = layer->getSize();
    for ( unsigned i = 0; i < n; ++i )
    {
        L.append( layer->getLb( i ) );
        U.append( layer->getUb( i ) );
    }
    lowerBounds = L;   // assign into caller-provided containers
    upperBounds = U;
}

unsigned DependencyAnalyzer::_argmaxAbsNonzero( const Vector<double> &w ) const
{
    ASSERT( w.size() > 0 );
    unsigned k = 0;
    double best = 0.0;
    for ( unsigned j = 0; j < w.size(); ++j )
    {
        double a = std::fabs( w[j] );
        if ( FloatUtils::isZero( a ) ) continue;
        if ( a > best ) { best = a; k = j; }
    }
    ASSERT( !FloatUtils::isZero( best ) ); // must have a nonzero pivot
    return k;
}

void DependencyAnalyzer::_boxMinMax( const Vector<double> &a, double b,
                                     const Vector<double> &L, const Vector<double> &U,
                                     double &outMin, double &outMax ) const
{
    ASSERT( a.size() == L.size() && a.size() == U.size() );
    double mn = b, mx = b;
    for ( unsigned j = 0; j < a.size(); ++j )
    {
        double aj = a[j];
        if ( aj >= 0 ) { mn += aj * L[j]; mx += aj * U[j]; }
        else           { mn += aj * U[j]; mx += aj * L[j]; }
    }
    outMin = mn; outMax = mx;
}

void DependencyAnalyzer::_sliceMinMax_givenOtherZero( const Vector<double> &w_t, double b_t,
                                                      const Vector<double> &w_o, double b_o,
                                                      const Vector<double> &L, const Vector<double> &U,
                                                      double &outMin, double &outMax ) const
{
    ASSERT( w_t.size() == w_o.size() && w_t.size() == L.size() && L.size() == U.size() );

    // pivot on largest-magnitude nonzero in w_o
    const unsigned k = _argmaxAbsNonzero( w_o );
    const double denom = w_o[k];
    ASSERT( !FloatUtils::isZero( denom ) );

    // eliminate x_k using w_o·x + b_o = 0  =>  x_k = -(b_o + sum_{j!=k} w_o[j] x_j) / w_o[k]
    // substitute into w_t·x + b_t  ==>  new affine in remaining variables: a·x + b
    Vector<double> a; 
    double b = b_t;

    for ( unsigned j = 0; j < w_t.size(); ++j )
    {
        if ( j == k ) continue;
        // coefficient of x_j after substitution:
        // w_t[j] + w_t[k] * ( w_o[j] / (-w_o[k]) ) = w_t[j] - w_t[k] * (w_o[j]/w_o[k])
        double coeff = w_t[j] - ( w_t[k] * ( w_o[j] / denom ) );
        a.append( coeff );
    }
    // constant term adds: w_t[k] * ( b_o / denom ) with a minus sign (since x_k = -(b_o + ...)/w_o[k])
    b = b_t - ( w_t[k] * ( b_o / denom ) );

    // Build reduced box (drop coordinate k)
    Vector<double> Lr, Ur;
    for ( unsigned j = 0; j < L.size(); ++j )
        if ( j != k ) { Lr.append( L[j] ); Ur.append( U[j] ); }

    // Box min/max on reduced form
    _boxMinMax( a, b, Lr, Ur, outMin, outMax );
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
