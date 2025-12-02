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
#include <unordered_set>
#include <algorithm>

DependencyAnalyzer::DependencyAnalyzer( const InputQuery *baseIpq,
                                        const Vector<Vector<double>> &allLbs,
                                        const Vector<Vector<double>> &allUbs )
    : _baseIpq( baseIpq )
    , _originalLbs( allLbs )
    , _originalUbs( allUbs )
{
    _numQueries = _originalLbs.size();
    ASSERT( _numQueries > 0 );

    _inputDim = _originalLbs[0].size();
    for ( unsigned q = 0; q < _numQueries; ++q )
        ASSERT( _originalLbs[q].size() == _inputDim &&
                _originalUbs[q].size() == _inputDim );

    _nextQueryToSolve = 0;
    ASSERT( _nextQueryToSolve == 0 );

    _computeCoveringBoxFromRemainingQueries();

    _context = nullptr;
    _seenPhase = nullptr;

    buildFromBase();
    std::printf("[DA] initial _baseIpq: vars=%u, eqs=%u\n",
        _baseIpq->getNumberOfVariables(),
        _baseIpq->getNumberOfEquations());

    _collectAllUnstableNeurons();
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
    // debugdiff()
}

DependencyAnalyzer::~DependencyAnalyzer() = default;

void DependencyAnalyzer::setContext( CVC4::context::Context *ctx )
{
    ASSERT( ctx );
    _context = ctx;


    unsigned numTightened = runBoundTightening();
    printf("[DA] DeepPoly tightening: %u tightenings\n", numTightened);

    //TODO: notify about stable neurons, those neurons might activate dependencies learned from prevous query, when testing assert that the learned implcation -active or inactive- was not done to see if this works

    computeSameLayerDependencies();

    // (Re)build runtime states for all currently-known dependencies
    _dependencyStates.clear(); //TODO: asster this - done before 
    _dependencyStates.reserve( _dependencies.size() );

    for ( unsigned id = 0; id < _dependencies.size(); ++id )
    {
        const Dependency &d = _dependencies[id];
        _addDependencyRuntimeState( id, d );
    }

    ASSERT( _dependencyStates.size() == _dependencies.size() );

    // Build context-dependent seen-phase map
    _seenPhase = new (true) CVC4::context::CDHashMap<unsigned, ReLURuntimeState, std::hash<unsigned>> ( _context );
}

unsigned DependencyAnalyzer::computeSameLayerDependencies()
{
    ASSERT( _networkLevelReasoner );

    const unsigned numLayers = _networkLevelReasoner->getNumberOfLayers();
    unsigned added = 0;
    unsigned totalAdded = 0;

    printf("[DA][debug] computeDependencies: scanning %u layers for weighted-sum layers\n",
           numLayers);

    for ( unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex )
    {
        const NLR::Layer *layer = _networkLevelReasoner->getLayer( layerIndex );
        if ( !layer )
            continue;

        const auto layerType = layer->getLayerType();

        // Adjust the enum constant name to match your codebase if needed
        if ( layerType == NLR::Layer::WEIGHTED_SUM ) //NLR::Layer::
        {
            printf("[DA][debug]   computing same-layer dependencies for weighted-sum layer %u\n",
                   layerIndex);
            added = computeSameLayerDependencies( layerIndex );
            totalAdded += added;
        }
    }

    printf("[DA][debug] computeDependencies: done\n");
    return totalAdded;
}
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
    unsigned numTightened = _applyTighteningsToPreprocessedQuery( tightenings );

    return numTightened;
}

unsigned DependencyAnalyzer::_applyTighteningsToPreprocessedQuery( const List<Tightening> &tightenings )
{
    if ( !_preprocessedQuery )
    {
        printf("[DA][error] applyTighteningsToPreprocessedQuery called but _preprocessedQuery is null\n");
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
            "applyTighteningsToPreprocessedQuery called before buildFromBase()" );
    }

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
        else  // UB
        {
            if ( FloatUtils::lt( x, _preprocessedQuery->getUpperBound( v ) ) )
            {
                _preprocessedQuery->setUpperBound( v, x );
                ++numTightened;
            }
        }
    }
    printf("[DA][_applyTighteningsToPreprocessedQuery] tightening: %u tightenings\n", numTightened);
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
            ASSERT( false );
        }

        //################### END DEBUGING ########################

        if ( lowerPreActivation < 0.0 && upperPreActivation > 0.0 )
            unstableNeurons.push_back( neuronIndex );
    }

    printf("[DA] Layer %u: found %zu unstable neurons\n", layerIndex, unstableNeurons.size());
    printf("[DA] unstable list: [");
    for ( unsigned i = 0; i < unstableNeurons.size(); ++i )
    {
        unsigned varI = weightedSumLayer->neuronToVariable( unstableNeurons[i] );
        printf("%u", varI);
        if ( i + 1 < unstableNeurons.size() ) printf(", ");
    }
    printf("]\n");

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
    if ( unstable.size() < 2 ) 
    {
        printf("[DA] not enough unstable neurons: %zu .\n", unstable.size());
        return 0;
    }
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

    return added;
}


bool DependencyAnalyzer::detectAndRecordPairConflict(unsigned layerIndex,
                                 unsigned q, unsigned r)
{

    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( layerIndex );
    unsigned varQ_ = weightedSumLayer->neuronToVariable( q );
    unsigned varR_ = weightedSumLayer->neuronToVariable( r );
    std::vector<unsigned> vars = { varQ_, varR_ };
    if (_isSupersetOfKnownDependency(vars))
        return false;

    Dependency d;
    if (!analyzePairConflict(layerIndex, q, r, d))
        return false;
    return recordConflict(std::move(d));
}

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

    if ( countTrue > 1 )
    {
        unsigned varQ_ = weightedSumLayer->neuronToVariable( q );
        unsigned varR_ = weightedSumLayer->neuronToVariable( r );

        printf("[DA][pair %u,%u] (vars %u,%u) >0(l_q_r0)=%d  <0(u_q_r0)=%d  "
               ">0(l_r_q0)=%d  <0(u_r_q0)=%d  totalTrue=%u\n",
               q, r, varQ_, varR_,
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

bool DependencyAnalyzer::_isSupersetOfKnownDependency(const std::vector<unsigned> &variables) const
{
    std::unordered_map<DependencyState::DependencyId, unsigned> dependencyOverlapCount;

    printf("[DA] Checking if candidate with variables { ");
    for (unsigned var : variables)
        printf("%u ", var);
    printf("} is a superset of any known dependency...\n");

    for (unsigned variable : variables)
    {
        const auto activeIt = _watchActive.find(variable);
        const auto inactiveIt = _watchInactive.find(variable);

        // Optional sanity check: ensure no overlap between active and inactive watchers
        if (activeIt != _watchActive.end() && inactiveIt != _watchInactive.end())
        {
            const auto &activeList = activeIt->second;
            const auto &inactiveList = inactiveIt->second;

            std::unordered_set<DependencyState::DependencyId> activeSet(activeList.begin(), activeList.end());
            for (DependencyState::DependencyId depId : inactiveList)
            {
                ASSERT(activeSet.find(depId) == activeSet.end());
            }
        }
        //TODO: delete this check before release

        if (activeIt != _watchActive.end())
        {
            printf("[DA] Variable %u has %u active dependencies\n", variable, activeIt->second.size());
            for (DependencyState::DependencyId depId : activeIt->second)
            {
                ++dependencyOverlapCount[depId];
                printf("    [DA] Active dep %u now has count %u\n", depId, dependencyOverlapCount[depId]);
            }
        }

        if (inactiveIt != _watchInactive.end())
        {
            printf("[DA] Variable %u has %u inactive dependencies\n", variable, inactiveIt->second.size());
            for (DependencyState::DependencyId depId : inactiveIt->second)
            {
                ++dependencyOverlapCount[depId];
                printf("    [DA] Inactive dep %u now has count %u\n", depId, dependencyOverlapCount[depId]);
            }
        }
    }

    for (const auto &[depId, count] : dependencyOverlapCount)
    {
        unsigned depSize = _dependencies[depId].size();
        printf("[DA] Checking dep %u: count %u vs size %u\n", depId, count, depSize);
        if (count == depSize)
        {
            printf("[DA] → Candidate is a superset of known dependency %u, skipping.\n", depId);
            return true;
        }
    }

    printf("[DA] → Candidate is not a superset of any known dependency.\n");
    return false;
}



bool DependencyAnalyzer::recordConflict( Dependency d )
{
    // --- Basic sanity checks ---
    ASSERT( d.size() >= 2 );
    ASSERT( d.getVars().size() == d.getStates().size() );

    // --- Get vars and states separately ---
    const std::vector<unsigned> &vars   = d.getVars();
    const std::vector<ReLUState> &states = d.getStates();

    // --- Canonical ordering sanity check ---
    for ( size_t i = 1; i < vars.size(); ++i )
    {
        ASSERT( vars[i - 1] < vars[i] );  // must be strictly ascending
    }
    // --- Debug-only duplicate assertion ---
    ASSERT( _dependencyIndex.find( d ) == _dependencyIndex.end() );  // should not already exist

    // --- Store dependency & create runtime state ---
    const DependencyState::DependencyId id = _addDependency( d );
    if ( _context )
        _addDependencyRuntimeState( id, d );

    // --- Register watches for each literal ---
    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned    v = vars[i];
        const ReLUState   s = states[i];

        // Choose bucket by polarity
        auto &bucket = ( s == ReLUState::Active ) ? _watchActive[v] : _watchInactive[v];

        // Debug Sanity: no duplicate ids in the same bucket
        for ( unsigned k = 0; k < bucket.size(); ++k )
            ASSERT( bucket[k] != id );
        // End Debug Sanity: no duplicate ids in the same bucket

        bucket.append( id );
    }

    return true; // Return true if newly inserted 
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

bool DependencyAnalyzer::notifyNeuronFixed( unsigned var, ReLUState state )
{
    ASSERT( _seenPhase );

    // Allow: (1) originally unstable vars, OR
    //        (2) stable var that become exactly at zero.
    //
    // Case (2):
    //    state = Inactive  AND ub == 0
    //    state = Active    AND lb == 0
    //
    double lb = _preprocessedQuery->getLowerBound( var );
    double ub = _preprocessedQuery->getUpperBound( var );

    bool originallyUnstable = _isUnstableVar( var );

    printf(
        "[DA][debug] notifyNeuronFixed(var=%u, state=%s) "
        "inactive&lb==0=%d  active&ub==0=%d  originallyUnstable=%d  bounds=[%.10g, %.10g]\n",
        var,
        (state == ReLUState::Inactive ? "Inactive" : "Active"),
        ( state == ReLUState::Inactive && FloatUtils::isZero( lb ) ),
        ( state == ReLUState::Active   && FloatUtils::isZero( ub ) ),
        originallyUnstable,
        lb, ub
    );

    bool allowed =
           originallyUnstable
        || ( state == ReLUState::Inactive && FloatUtils::isZero( lb ) )
        || ( state == ReLUState::Active   && FloatUtils::isZero( ub ) );

    if ( !allowed ) // TOOD: turn into assertion
    {
        printf(
            "[DA][warn] notifyNeuronFixed on var=%u, state=%s but var is not in "
            "_unstableNeurons and not a zero-edge case. bounds=[%.10g, %.10g]\n",
            var,
            (state == ReLUState::Inactive ? "Inactive" : "Active"),
            lb, ub
        );
        // Do NOT assert here for now; just let it continue.
    }

    // Map ReLUState -> runtime state enum
    const ReLURuntimeState incoming =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Active
                                       : ReLURuntimeState::Inactive;

    auto ait = _seenPhase->find( var );    
    ASSERT (ait == _seenPhase->end())
    _seenPhase->insert( var, incoming );

    /**************** Debug: print current _seenPhase map ****************/
    printf("[DA][debug] notifyNeuronFixed(var=%u, state=%s)\n",
           var, (state == ReLUState::Active ? "Active" : "Inactive"));

    if ( !_seenPhase->empty() )
    {
        printf("[DA][debug] _seenPhase contents:\n");
        for ( auto it = _seenPhase->begin(); it != _seenPhase->end(); ++it )
        {
            const unsigned trackedVar = (*it).first;
            const ReLURuntimeState rt = (*it).second;
            const char *phaseStr = "Unstable";
            if ( rt == ReLURuntimeState::Active )   phaseStr = "Active";
            else if ( rt == ReLURuntimeState::Inactive ) phaseStr = "Inactive";
            printf("   var=%u  -> %s\n", trackedVar, phaseStr);
        }
    }
    else
    {
        printf("[DA][debug] _seenPhase is empty.\n");
    }
    /**************** End Debug ****************/

    Vector<DependencyState::DependencyId> depsToUpdate;

    // Add watchers for (var, Active)
    {
        auto itA = _watchActive.find( var );
        if ( itA != _watchActive.end() )
        {
            const auto &vecA = itA->second;
            for ( unsigned i = 0; i < vecA.size(); ++i )
                depsToUpdate.append( vecA[i] );
        }
    }

    // Add watchers for (var, Inactive)
    {
        auto itI = _watchInactive.find( var );
        if ( itI != _watchInactive.end() )
        {
            const auto &vecI = itI->second;
            for ( unsigned i = 0; i < vecI.size(); ++i )
                depsToUpdate.append( vecI[i] );
        }
    }

    if ( depsToUpdate.empty() )
    {
        printf("[DA][debug] No dependencies watch var %u (either phase)\n", var);
        return false;
    }

    printf("[DA][debug] var %u is watched by %u dependencies\n",
        var, depsToUpdate.size());

    // IDs of dependencies that contain literal (var, state)
    const Vector<DependencyState::DependencyId> &depIds = depsToUpdate;
    bool foundDep = false;

    // Update runtime states for all dependencies watching (var, state)
    for ( unsigned t = 0; t < depIds.size(); ++t )
    {
        const DependencyState::DependencyId depId = depIds[t];
        ASSERT( depId < _dependencies.size() && depId < _dependencyStates.size() );

        const Dependency &dep = _dependencies[ depId ];
        DependencyState  &depState  = _dependencyStates[ depId ];

        // Find aligned index i s.t. dep.getVars()[i] == var
        const std::vector<unsigned>   &vars   = dep.getVars();
        // const std::vector<ReLUState>  &literalPhases = dep.getStates();  // polarity of the nogood (not used here)

        int idx = -1;
        for ( unsigned i = 0; i < vars.size(); ++i )
            if ( vars[i] == var ) { idx = i; break; }

        // Sanity: the var must be present (because it’s in the watch list)
        ASSERT( idx >= 0 );

        // Sanity: we should be transitioning from Unstable → {Active/Inactive}
        ASSERT( depState.getLiteralState( idx ) == ReLURuntimeState::Unstable );
        if ( state == ReLUState::Active )
            depState.setActive( idx );
        else
            depState.setInactive( idx );

        // Apply the observed runtime state
        unsigned impliedVar = 0;
        ReLUState impliedPhase = ReLUState::Active; // default, will be overwritten

        if ( depState.checkImplication( dep, impliedVar, impliedPhase ) )
        {
            // For now we just record that this dependency is “active”;
            // later we can also store (impliedVar, impliedPhase) somewhere.
            // TODO: check that it is not in _activeDepIds before appending

            // Add depId only if not already present
            if ( std::find( _activeDepIds.begin(), _activeDepIds.end(), depId )
                == _activeDepIds.end() )
            {
                _activeDepIds.append( depId );
                foundDep = true;
                printf("[DA] Dep %u added to _activeDepIds\n", depId);
            }
            else
            {
                continue;
                printf("[DA] Dep %u already present in _activeDepIds, skipping\n", depId);
            }

            // Optional debug:
            printf("[DA] Dep %u implies var %u must be %s\n",
                depId,
                impliedVar,
                (impliedPhase == ReLUState::Active ? "Active" : "Inactive"));
        }


    }
    if ( foundDep )
    {
        printf("Found!!");
        printf("[DA] [notifyNeuronFixed] Applicable dependecies found: %u \n", _activeDepIds.size());
    }
    return foundDep;
}

void DependencyAnalyzer::notifyLowerBoundUpdate( unsigned variable,
                                                 double previousLowerBound,
                                                 double newLowerBound )
{
    // Lower bounds must only move up (monotone tightening)
    if ( !FloatUtils::gt( newLowerBound, previousLowerBound ) )
        return;
    ASSERT( !FloatUtils::lt( newLowerBound, previousLowerBound ) );

    // Detect transition to guaranteed Active
    printf("[DA] LB %u for %.6g -> %.6g\n", variable, previousLowerBound, newLowerBound);
    if ( previousLowerBound <= 0.0 && newLowerBound >= 0.0 ){
        printf("[DA] Valid LB\n");
        notifyNeuronFixed( variable, ReLUState::Active );}
}

void DependencyAnalyzer::notifyUpperBoundUpdate( unsigned variable,
                                                 double previousUpperBound,
                                                 double newUpperBound )
{
    // Upper bounds must only move down (monotone tightening)
    if ( !FloatUtils::lt( newUpperBound, previousUpperBound ) )
        return;
    ASSERT( !FloatUtils::gt( newUpperBound, previousUpperBound ) );

    // Detect transition to guaranteed Inactive
    printf("[DA] UB %u for %.6g -> %.6g\n", variable, previousUpperBound, newUpperBound);
    if ( previousUpperBound >= 0.0 && newUpperBound <= 0.0 ){
        printf("[DA] Valid UB\n");
        notifyNeuronFixed( variable, ReLUState::Inactive );}
}

DependencyState::DependencyId DependencyAnalyzer::_addDependency( const Dependency &d )
{
    const DependencyState::DependencyId id = _dependencies.size();
    _dependencies.push_back( d );

    // Track for future duplicate ASSERTs
    _dependencyIndex.emplace( d, id );

    return id;
}

void DependencyAnalyzer::_addDependencyRuntimeState( DependencyState::DependencyId id,
                                                    const Dependency &d )
{
    ASSERT( _context ); // make sure setContext() was called

    DependencyState st( id,
                        static_cast<unsigned>( d.size() ),
                        *_context );
    _dependencyStates.push_back( std::move( st ) );

    /**************** For Debugging ********************/
    const DependencyState &last = _dependencyStates.back();
    printf("[DA] Added DependencyState id=%u, size=%u | total now=%zu\n",
           id,
           static_cast<unsigned>( d.size() ),
           _dependencyStates.size() );

    printf("   [%u] depId=%u  literals=%u  pattern=",
        id, last.getDepId(), last.size() );

    const std::vector<unsigned>   &vars   = d.getVars();
    const std::vector<ReLUState>  &phases = d.getStates();

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        printf(" (%u,%c)", 
            vars[i],
            phases[i] == ReLUState::Active ? 'A' : 'I');
    }

    printf("\n");

    /**************** End For Debugging ********************/
}


void DependencyAnalyzer::_computeCoveringBoxFromRemainingQueries()
{
    ASSERT( _nextQueryToSolve < _numQueries );

    _currentLb = Vector<double>( _inputDim, +INFINITY );
    _currentUb = Vector<double>( _inputDim, -INFINITY );

    for ( unsigned x = 0; x < _inputDim; ++x )
    {
        double lb = +INFINITY;
        double ub = -INFINITY;

        for ( unsigned q = _nextQueryToSolve; q < _numQueries; ++q )
        {
            lb = std::min( lb, _originalLbs[q][x] );
            ub = std::max( ub, _originalUbs[q][x] );
        }

        _currentLb[x] = lb;
        _currentUb[x] = ub;
    }
}

bool DependencyAnalyzer::_isSubset( const Vector<double> &lbNew,
                                   const Vector<double> &ubNew,
                                   const Vector<double> &lbOld,
                                   const Vector<double> &ubOld ) const

{
    for ( unsigned x = 0; x < _inputDim; ++x )
    {
        if ( lbNew[x] < lbOld[x] ) return false;
        if ( ubNew[x] > ubOld[x] ) return false;
    }
    return true;
}

void DependencyAnalyzer::notifyQuerySolved()
{
    // todo cleaning and preparing for the new context and query.
    ASSERT( _nextQueryToSolve < _numQueries);
    ++_nextQueryToSolve;

    if ( _nextQueryToSolve >= _numQueries )
        return;

    Vector<double> oldLb = _currentLb;
    Vector<double> oldUb = _currentUb;

    _computeCoveringBoxFromRemainingQueries();

    // === Debug info ===
    printf("[DA][debug] notifyQuerySolved(): tightening input domain\n");

    // unsigned changedCount = 0;
    // double maxDelta = 0.0;

    // for ( unsigned i = 0; i < _inputDim; ++i )
    // {
    //     double oldL = oldLb[i];
    //     double oldU = oldUb[i];
    //     double newL = _currentLb[i];
    //     double newU = _currentUb[i];

    //     bool changed = (oldL != newL) || (oldU != newU);
    //     if ( changed )
    //     {
    //         ++changedCount;

    //         double dL = fabs(oldL - newL);
    //         double dU = fabs(oldU - newU);
    //         maxDelta = std::max({maxDelta, dL, dU});

    //         printf("[DA][debug]   dim %u: LB %.6f → %.6f (Δ=%.6f), "
    //                "UB %.6f → %.6f (Δ=%.6f)\n",
    //                i, oldL, newL, dL,
    //                   oldU, newU, dU);
    //     }
    // }

    // printf("[DA][debug]   total changed dims = %u / %u, max change = %.6f\n",
    //        changedCount, _inputDim, maxDelta);
    // === End of Debug info ===

    // --- Build tightenings for input variables based on the new covering box
    List<Tightening> inputTightenings;


    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( 0 );
    if ( !weightedSumLayer ) {
        printf("[DA] notifyQuerySolved: layer %u not found in NLR\n", 0);
        return;
    }
    const unsigned numNeurons = weightedSumLayer->getSize();
    ASSERT (_inputDim == numNeurons);
    for ( unsigned i = 0; i < _inputDim; ++i )
    {
        // Map input-dimension i to the corresponding variable in the preprocessed query.
        const unsigned var = weightedSumLayer->neuronToVariable( i );

        const double oldL = oldLb[i];
        const double oldU = oldUb[i];
        const double newL = _currentLb[i];
        const double newU = _currentUb[i];

        // Lower bound tightened?
        if ( FloatUtils::gt( newL, oldL ) )
        {
            Tightening tLB( var, newL, Tightening::LB );
            inputTightenings.append( tLB );
        }

        // Upper bound tightened?
        if ( FloatUtils::lt( newU, oldU ) )
        {
            Tightening tUB( var, newU, Tightening::UB );
            inputTightenings.append( tUB );
        }
    }

    if ( !inputTightenings.empty() )
    {
        unsigned numApplied = _applyTighteningsToPreprocessedQuery( inputTightenings );
        printf("[DA][debug]   applied %u input tightenings to preprocessed query\n", numApplied );
    }
    else
    {
        printf("[DA][debug]   no input tightenings to apply after notifyQuerySolved()\n");
    }
    ASSERT( _isSubset( _currentLb, _currentUb, oldLb, oldUb ) );
    _context = nullptr;
    _seenPhase = nullptr;
    _dependencyStates.clear();

    // computeDependencies(); # todo in the setcontex
}

void DependencyAnalyzer::getImpliedTightenings( List<Tightening> &tightenings )
{
    // Caller should pass an empty list; we only append.
    // Preconditions: setContext() was called, so these should be valid.
    ASSERT( _context );
    ASSERT( _preprocessedQuery );
    ASSERT( _networkLevelReasoner );

    if ( _activeDepIds.empty() )
    {
        printf("[DA][getImpliedTightenings] no active dependencies\n");
        return;
    }

    printf("[DA][getImpliedTightenings] processing %u active dependencies\n",
           _activeDepIds.size());

    for ( unsigned idx = 0; idx < _activeDepIds.size(); ++idx )
    {
        const DependencyState::DependencyId depId = _activeDepIds[idx];

        ASSERT( depId < _dependencies.size() );
        ASSERT( depId < _dependencyStates.size() );

        const Dependency &dep      = _dependencies[depId];
        DependencyState  &depState = _dependencyStates[depId];

        unsigned  impliedVar   = 0;
        ReLUState impliedPhase = ReLUState::Active; // will be overwritten

        bool hasImplication = depState.checkImplication( dep, impliedVar, impliedPhase );
        // By design, any depId in _activeDepIds must imply something.
        if( !hasImplication ) continue;

        double lb = _preprocessedQuery->getLowerBound( impliedVar );
        double ub = _preprocessedQuery->getUpperBound( impliedVar );

        printf("[DA][getImpliedTightenings] dep %u implies var %u must be %s "
               "(current bounds: [%.10g, %.10g])\n",
               depId,
               impliedVar,
               ( impliedPhase == ReLUState::Active ? "Active" : "Inactive" ),
               lb, ub );

        if ( impliedPhase == ReLUState::Active )
        {
            // Active ⇒ pre-activation >= 0
            const double newLb = 0.0;

            // Safety: new LB cannot exceed current UB
            ASSERT( !FloatUtils::gt( newLb, ub ) );
            ASSERT( FloatUtils::gt( newLb, lb ) );

            // If this does not strengthen the LB, skip emitting a tightening
            if ( !FloatUtils::gt( newLb, lb ) )
            {
                printf("[DA][getImpliedTightenings]   skip: LB already >= 0 for var %u\n",
                       impliedVar );
                continue;
            }

            Tightening t( impliedVar, newLb, Tightening::LB );
            tightenings.append( t );

            printf("[DA][getImpliedTightenings]   -> emit LB tightening: x%u >= %.10g\n",
                   impliedVar, newLb );
        }
        else
        {
            // impliedPhase == ReLUState::Inactive
            ASSERT( impliedPhase == ReLUState::Inactive );
            const double newUb = 0.0;

            // Safety: new UB cannot go below current LB
            ASSERT( !FloatUtils::lt( newUb, lb ) );
            ASSERT( FloatUtils::lt( newUb, ub ) );

            // If this does not strengthen the UB, skip
            if ( !FloatUtils::lt( newUb, ub ) )
            {
                printf("[DA][getImpliedTightenings]   skip: UB already <= 0 for var %u\n",
                       impliedVar );
                continue;
            }

            Tightening t( impliedVar, newUb, Tightening::UB );
            tightenings.append( t );

            printf("[DA][getImpliedTightenings]   -> emit UB tightening: x%u <= %.10g\n",
                   impliedVar, newUb );
        }
    }

    // Simple initial policy: after we’ve emitted tightenings for all active deps,
    // clear the list. New notifications will repopulate _activeDepIds.
    _activeDepIds.clear();
    printf("[DA][getImpliedTightenings] done, emitted %u tightenings\n",
           tightenings.size());
}

void DependencyAnalyzer::_collectAllUnstableNeurons()
{
    _unstableNeurons.clear();

    if ( !_networkLevelReasoner )
    {
        printf("[DA][_collectAllUnstableNeurons] NLR not set\n");
        return;
    }

    // Ensure NLR bounds are synced with the preprocessed query
    _networkLevelReasoner->obtainCurrentBounds( *_preprocessedQuery );

    const unsigned numLayers = _networkLevelReasoner->getNumberOfLayers();
    printf("[DA][_collectAllUnstableNeurons] scanning %u layers for unstable neurons\n",
           numLayers);

    for ( unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex )
    {
        const NLR::Layer *layer = _networkLevelReasoner->getLayer( layerIndex );
        if ( !layer )
            continue;

        const auto layerType = layer->getLayerType();

        if ( layerType == NLR::Layer::WEIGHTED_SUM )
        {
            printf("[DA][_collectAllUnstableNeurons]   layer %u is WEIGHTED_SUM\n",
                   layerIndex);

            std::vector<unsigned> unstableIndices;
            collectUnstableNeurons( layerIndex, unstableIndices );

            for ( unsigned neuronIndex : unstableIndices )
            {
                unsigned var = layer->neuronToVariable( neuronIndex );
                _unstableNeurons.push_back( var );
            }
        }
    }

    // Optional: deduplicate and sort
    std::sort( _unstableNeurons.begin(), _unstableNeurons.end() );
    _unstableNeurons.erase(
        std::unique( _unstableNeurons.begin(), _unstableNeurons.end() ),
        _unstableNeurons.end()
    );

    printf("[DA][_collectAllUnstableNeurons] total unstable vars = %zu\n",
           _unstableNeurons.size());
}

bool DependencyAnalyzer::_isUnstableVar( unsigned var ) const
{
    return std::binary_search( _unstableNeurons.begin(),
                               _unstableNeurons.end(),
                               var );
}

void DependencyAnalyzer::syncWithEnginePreprocessedQuery( const Query &engineQuery )
{
    ASSERT( _preprocessedQuery );
    ASSERT( _networkLevelReasoner );
    ASSERT( _context ); // setContext must have been called

    if ( _unstableNeurons.empty() )
    {
        printf("[DA][syncWithEnginePreprocessedQuery] no unstable neurons recorded\n");
        return;
    }

    printf("[DA][syncWithEnginePreprocessedQuery] syncing %zu unstable vars with Engine PQ\n",
           _unstableNeurons.size());

    for ( unsigned var : _unstableNeurons )
    {
        // Bounds from the Engine's preprocessed query
        const double lb = engineQuery.getLowerBound( var );
        const double ub = engineQuery.getUpperBound( var );

        printf("[DA][sync] var %u has Engine bounds [%.10g, %.10g]\n", var, lb, ub);

        // Already guaranteed Active?
        if ( !FloatUtils::lt( lb, 0.0 ) )
        {
            printf("[DA][sync] var %u is already Active (lb >= 0), notifying LB update\n", var);
            // Use -INFINITY as previous lower bound to force a "crossing zero" event
            notifyLowerBoundUpdate( var, -INFINITY, lb );
        }
        // Already guaranteed Inactive?
        else if ( !FloatUtils::gt( ub, 0.0 ) )
        {
            printf("[DA][sync] var %u is already Inactive (ub <= 0), notifying UB update\n", var);
            // Use +INFINITY as previous upper bound to force a "crossing zero" event
            notifyUpperBoundUpdate( var, +INFINITY, ub );
        }
        else
        {
            // Still unstable in Engine's view; do nothing now.
            printf("[DA][sync] var %u remains unstable in Engine PQ\n", var);
        }
    }
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
