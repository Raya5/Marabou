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
#include <chrono>
#include <unordered_map>
#include <cstdint>

#include <cmath>
#include <vector>
#include <cstdio>

#ifdef ENABLE_GUROBI
#include "GurobiWrapper.h"
#endif

DependencyAnalyzer::DependencyAnalyzer( const InputQuery *baseIpq,
                                        const Vector<Vector<double>> &allLbs,
                                        const Vector<Vector<double>> &allUbs )
    : _baseIpq( baseIpq )
    , _originalLbs( allLbs )
    , _originalUbs( allUbs )
{
    // Store the batch bounds and basic dimensions
    _numQueries = _originalLbs.size();
    ASSERT( _numQueries > 0 );

    _inputDim = _originalLbs[0].size();

    // Sanity: each query must have the same input dimension
    for ( unsigned q = 0; q < _numQueries; ++q )
        ASSERT( _originalLbs[q].size() == _inputDim &&
                _originalUbs[q].size() == _inputDim );

    // Start from the first query in the batch
    _nextQueryToSolve = 0;
    ASSERT( _nextQueryToSolve == 0 );

    // Build the initial "covering box" over all remaining queries
    _computeCoveringBoxFromRemainingQueries();

    // These are injected later from Engine
    _context = nullptr;
    _preprocessor = nullptr;
    _currPreprocessedQuery = nullptr;
    _currNetworkLevelReasoner = nullptr;   
    _seenPhase = nullptr;

    // Preprocess the base query and cache its NLR
    buildFromBase();

    // Collect all ReLUs that start unstable (used for SAT mapping/bitmask sizing)
    _collectAllUnstableNeurons();
    _bitmaskSize = _unstableNeurons.size();

    // Initialize CaDiCaL mapping tables (no clauses yet)
    _initializeSatSolver();
}

void DependencyAnalyzer::buildFromBase()
{
    // Base query must exist
    if ( !_baseIpq )
    {
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "DependencyAnalyzer::buildFromBase called with null baseIpq." );
    }

    // Preprocess a private copy of the base query (variable elimination enabled)
    _preprocessedQuery = _baseIpqPreprocessor.preprocess(
        *_baseIpq, GlobalConfiguration::PREPROCESSOR_ELIMINATE_VARIABLES );

    // Cache the NLR owned by the preprocessed query
    _networkLevelReasoner =
        _preprocessedQuery ? _preprocessedQuery->getNetworkLevelReasoner() : nullptr;

    // Must have an NLR in order to run DeepPoly and read layers/weights
    if ( !_networkLevelReasoner )
    {
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Preprocessing failed: NetworkLevelReasoner is null." );
    }

    // Ensure successor layer metadata is computed (needed by some NLR routines)
    _networkLevelReasoner->computeSuccessorLayers();
}

DependencyAnalyzer::~DependencyAnalyzer() = default;

void DependencyAnalyzer::setContext( CVC4::context::Context *ctx )
{
    // Engine provides a context per solve; we store it for CD* datastructures
    ASSERT( ctx );
    _context = ctx;

    // Run NLR tightening on the analyzer's preprocessed query once per context
    (void)runBoundTightening();

    // Compute dependencies (nogoods) based on the tightened bounds
    // computeSameLayerDependencies();

    rebuildDependencyRuntimeStates();

    // Track seen ReLU phases in a context-dependent map (backtrackable)
    _seenPhase =
        new (true) CVC4::context::CDHashMap<unsigned, ReLURuntimeState, std::hash<unsigned>>( _context );
}

void DependencyAnalyzer::rebuildDependencyRuntimeStates()
{
    // Rebuild dependency runtime state objects under the current context
    _dependencyStates.clear();
    _dependencyStates.reserve( _dependencies.size() );

    for ( unsigned id = 0; id < _dependencies.size(); ++id )
    {
        const Dependency &d = _dependencies[id];
        _addDependencyRuntimeState( id, d );
    }

    ASSERT( _dependencyStates.size() == _dependencies.size() );
}


void DependencyAnalyzer::setPreprocessor( Preprocessor *preprocessor )
{
    // Engine provides the preprocessor so we can map old<->new indices
    _preprocessor = preprocessor;
}

void DependencyAnalyzer::setCurrentPreprocessedQuery( const Query &enginePreprocessedQuery )
{
    // Make an owned copy (so DA is not tied to Engine's lifetime/mutations)
    _currPreprocessedQuery = std::make_unique<Query>( enginePreprocessedQuery );
}

const Query *DependencyAnalyzer::getCurrentPreprocessedQuery() const
{
    return _currPreprocessedQuery.get();
}


unsigned DependencyAnalyzer::computeSameLayerDependencies()
{
    // Scan all layers and compute dependencies only for WEIGHTED_SUM layers
    ASSERT( _networkLevelReasoner );
    printf("[Debug] Computing same-layer dependencies...\n");

    const unsigned numLayers = _networkLevelReasoner->getNumberOfLayers();
    unsigned totalAdded = 0;

    for ( unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex )
    {
        const NLR::Layer *layer = _networkLevelReasoner->getLayer( layerIndex );
        if ( !layer )
            continue;

        const auto layerType = layer->getLayerType();

        if ( layerType == NLR::Layer::WEIGHTED_SUM )
            totalAdded += computeSameLayerDependencies( layerIndex );
    }

    printf("[Debug] Computed %u same-layer dependencies.\n", totalAdded );
    return totalAdded;
}

const InputQuery *DependencyAnalyzer::getBaseInputQuery() const
{
    return _baseIpq;
}

void DependencyAnalyzer::printSummary() const
{
    // Intentionally no-op in clean mode (used to print diagnostic summary).
}

unsigned DependencyAnalyzer::runBoundTightening()
{
    // Tightening requires a preprocessed query and NLR
    if ( !_preprocessedQuery || !_networkLevelReasoner )
    {
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "runBoundTightening called before buildFromBase()" );
    }

    // Run DeepPoly on current bounds stored in the NLR/query
    _networkLevelReasoner->deepPolyPropagation();

    // Collect proposed tightenings from NLR constraints
    List<Tightening> tightenings;
    _networkLevelReasoner->getConstraintTightenings( tightenings );

    // Apply tightenings back to the analyzer's preprocessed query (Engine-like)
    return _applyTighteningsToPreprocessedQuery( tightenings );
}

unsigned DependencyAnalyzer::_applyTighteningsToPreprocessedQuery( const List<Tightening> &tightenings )
{
    // Must have a preprocessed query to write into
    if ( !_preprocessedQuery )
    {
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "applyTighteningsToPreprocessedQuery called before buildFromBase()" );
    }

    unsigned numTightened = 0;

    // Apply only strengthening bounds (monotone)
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

    return numTightened;
}

void DependencyAnalyzer::collectUnstableNeurons( unsigned layerIndex,
                                                bool currQuery,
                                                std::vector<unsigned> &unstableNeurons ) const
{
    // Collect neuron indices in this layer whose pre-activation interval crosses 0
    unstableNeurons.clear();
    const Query *query = currQuery ? _currPreprocessedQuery.get() : _preprocessedQuery.get();
    ASSERT( query );

    const NLR::NetworkLevelReasoner *networkLevelReasoner = currQuery ? _currNetworkLevelReasoner : _networkLevelReasoner;
    if ( !networkLevelReasoner )
        ASSERT( false && "NLR is null in collectUnstableNeurons" );

    const NLR::Layer *weightedSumLayer = networkLevelReasoner->getLayer( layerIndex );
    if ( !weightedSumLayer )
        ASSERT( false && "Layer is null in collectUnstableNeurons" );

    const unsigned numNeurons = weightedSumLayer->getSize();
    for ( unsigned neuronIndex = 0; neuronIndex < numNeurons; ++neuronIndex )
    {

        // Consistency check: NLR bounds should match query bounds (kept as ASSERT)
        const unsigned var = weightedSumLayer->neuronToVariable( neuronIndex );
        const double nlrLb = weightedSumLayer->getLb( neuronIndex );
        const double pqLb  = query->getLowerBound( var );
        const double nlrUb = weightedSumLayer->getUb( neuronIndex );
        const double pqUb  = query->getUpperBound( var );

        if ( FloatUtils::lt( nlrLb, pqLb ) ||
             FloatUtils::gt( nlrUb, pqUb ) )
        {
            printf( "[DA][Warning] NLR/query bound mismatch for var %u: NLR=[%g,%g], PQ=[%g,%g]\n",
                    var, nlrLb, nlrUb, pqLb, pqUb );
            ASSERT( false );
        }

        const double lowerPreActivation = nlrLb;
        const double upperPreActivation = nlrUb;

        // Unstable means: can be negative or positive depending on input
        if ( lowerPreActivation < 0.0 && upperPreActivation > 0.0 )
            unstableNeurons.push_back( neuronIndex );
    }
}

unsigned DependencyAnalyzer::computeSameLayerDependencies( unsigned weightedSumLayerIndex )
{
    // For each weighted-sum layer, enumerate combinations of unstable neurons
    // and detect "conflicts" (forbidden phase patterns).
    if ( !_networkLevelReasoner )
        return 0;

    // Sync NLR's internal bounds with the analyzer query
    _networkLevelReasoner->obtainCurrentBounds( *_preprocessedQuery );

    const auto *weightedSumLayer = _networkLevelReasoner->getLayer( weightedSumLayerIndex );
    if ( !weightedSumLayer || weightedSumLayer->getLayerType() != NLR::Layer::WEIGHTED_SUM )
        return 0;

    std::vector<unsigned> unstable;
    collectUnstableNeurons( weightedSumLayerIndex, true, unstable);
    const unsigned before = unstable.size();
    // _pruneUnstableByTopKWithRecency( weightedSumLayerIndex,
    //                                 unstable,
    //                                 /*fractionToKeep=*/0.25,
    //                                 /*minK=*/16,
    //                                 /*maxK=*/32 );
    const unsigned after = unstable.size();

    printf( "[DA][score] layer=%u unstable before=%u after=%u\n",
            weightedSumLayerIndex, before, after );


    // Need at least 2 vars to form a dependency
    if ( unstable.size() < 2 )
        return 0;
    
    // Sort for consistent ordering
    std::sort( unstable.begin(), unstable.end() );

    unsigned addedPairs   = 0;
    unsigned addedTriples = 0;
    unsigned addedQuads   = 0;

    // Enumerate all unordered pairs
    for ( size_t i = 0; i + 1 < unstable.size(); ++i )
    {
        const unsigned q = unstable[i];
        for ( size_t j = i + 1; j < unstable.size(); ++j )
        {
            const unsigned r = unstable[j];
            ASSERT( q < r );
            if ( detectAndRecordConflict( weightedSumLayerIndex, { q, r } ) )
                ++addedPairs;
        }
    }

    // Enumerate all unordered triples (if any)
    // if ( unstable.size() >= 3 )
    // {
    //     for ( size_t i = 0; i + 2 < unstable.size(); ++i )
    //     {
    //         const unsigned q = unstable[i];
    //         for ( size_t j = i + 1; j + 1 < unstable.size(); ++j )
    //         {
    //             const unsigned r = unstable[j];
    //             for ( size_t k = j + 1; k < unstable.size(); ++k )
    //             {
    //                 const unsigned s = unstable[k];
    //                 ASSERT( q < r && r < s );
    //                 if ( detectAndRecordConflict( weightedSumLayerIndex, { q, r, s } ) )
    //                     ++addedTriples;
    //             }
    //         }
    //     }
    // }

    // Enumerate all unordered quadruples (if any)
    // if ( unstable.size() >= 4 )
    // {
    //     for ( size_t i = 0; i + 3 < unstable.size(); ++i )
    //     {
    //         const unsigned q = unstable[i];
    //         for ( size_t j = i + 1; j + 2 < unstable.size(); ++j )
    //         {
    //             const unsigned r = unstable[j];
    //             for ( size_t k = j + 1; k + 1 < unstable.size(); ++k )
    //             {
    //                 const unsigned s = unstable[k];
    //                 for ( size_t m = k + 1; m < unstable.size(); ++m )
    //                 {
    //                     const unsigned t = unstable[m];
    //                     ASSERT( q < r && r < s && s < t );
    //                     if ( detectAndRecordConflict( weightedSumLayerIndex, { q, r, s, t } ) )
    //                         ++addedQuads;
    //                 }
    //             }
    //         }
    //     }
    // }

    return addedPairs + addedTriples + addedQuads;
}

bool DependencyAnalyzer::analyzeConflict( unsigned layerIndex,
                                         const std::vector<unsigned> &neurons,
                                         Dependency &outDependency )
{
    // Analyze a specific set of neurons and determine if their phases form a forbidden pattern.
    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( layerIndex );
    ASSERT( weightedSumLayer );
    ASSERT( weightedSumLayer->getLayerType() == NLR::Layer::WEIGHTED_SUM );

    const unsigned k = neurons.size();
    ASSERT( k == 2 || k == 3 || k == 4 );

    // Validate ordering and bounds
    const unsigned layerSize = weightedSumLayer->getSize();
    for ( unsigned i = 0; i < k; ++i )
    {
        ASSERT( neurons[i] < layerSize );
        if ( i > 0 ) ASSERT( neurons[i - 1] < neurons[i] );
    }
    (void)layerSize;

    // This implementation assumes a single predecessor (typical affine layer)
    const auto &sources = weightedSumLayer->getSourceLayers();
    ASSERT( sources.size() == 1 );
    const unsigned prevLayerIndex = sources.begin()->first;
    const NLR::Layer *prevLayer = _networkLevelReasoner->getLayer( prevLayerIndex );
    ASSERT( prevLayer );
    const unsigned prevSize = prevLayer->getSize();

    // Bounds on previous layer variables
    Vector<double> lowerPrev, upperPrev;
    _getLayerBounds( prevLayer, lowerPrev, upperPrev );

    // Collect weights and biases for the neurons under test
    std::vector<Vector<double>> W;
    std::vector<double> B;
    W.reserve( k );
    B.reserve( k );

    for ( unsigned idx = 0; idx < k; ++idx )
    {
        const unsigned n = neurons[idx];
        Vector<double> w( prevSize );
        for ( unsigned j = 0; j < prevSize; ++j )
            w[j] = weightedSumLayer->getWeight( prevLayerIndex, j, n );

        W.push_back( w );
        B.push_back( weightedSumLayer->getBias( n ) );
    }

    // For each neuron i, compute its min/max when the other neurons are constrained to 0
    std::vector<double> Lcond( k, 0.0 ), Ucond( k, 0.0 );

    for ( unsigned i = 0; i < k; ++i )
    {
        if ( k == 2 )
        {
            // Size-2 supports an analytic slicing approximation
            const unsigned o = ( i == 0 ? 1 : 0 );
            _sliceMinMax_givenOtherZero( W[i], B[i], W[o], B[o],
                                         lowerPrev, upperPrev,
                                         Lcond[i], Ucond[i] );
        }
        else
        {
            // For k>=3, we rely on an LP slice (Gurobi) to enforce multiple equalities
#ifdef ENABLE_GUROBI
            Vector<Vector<double>> w_eq( k - 1 );
            Vector<double> b_eq( k - 1 );

            unsigned p = 0;
            for ( unsigned j = 0; j < k; ++j )
            {
                if ( j == i ) continue;
                w_eq[p] = W[j];
                b_eq[p] = B[j];
                ++p;
            }

            _sliceMinMax_givenMEqZero_LP( W[i], B[i],
                                          w_eq, b_eq,
                                          lowerPrev, upperPrev,
                                          Lcond[i], Ucond[i] );
#else
            ASSERT( false && "analyzeConflict size>=3 requires ENABLE_GUROBI" );
#endif
        }
    }

    // If any slice is infeasible, ignore the conflict
    for ( unsigned i = 0; i < k; ++i )
    {
        if ( !FloatUtils::isFinite( Lcond[i] ) || !FloatUtils::isFinite( Ucond[i] ) )
            return false;
    }

    // Determine forced phases from the sliced bounds:
    //  - forced Active if Lcond > 0
    //  - forced Inactive if Ucond < 0
    std::vector<bool> forcedActive( k, false ), forcedInactive( k, false );
    for ( unsigned i = 0; i < k; ++i )
    {
        forcedActive[i]   = FloatUtils::gt( Lcond[i], 0.0 );
        forcedInactive[i] = FloatUtils::lt( Ucond[i], 0.0 );
    }

    auto isForced = []( bool fa, bool fi ) {
        // Exactly one of the two must be true
        return ( fa && !fi ) || ( fi && !fa );
    };

    // Require every neuron in the set to be forced to a unique phase
    for ( unsigned i = 0; i < k; ++i )
        if ( !isForced( forcedActive[i], forcedInactive[i] ) )
            return false;

    // Build the forbidden phase assignment ("nogood"):
    // If neuron i is forced Inactive, then the forbidden pattern sets it Active (and vice versa).
    std::vector<ReLUState> forbid( k, ReLUState::Inactive );
    for ( unsigned i = 0; i < k; ++i )
        forbid[i] = forcedInactive[i] ? ReLUState::Active : ReLUState::Inactive;

    // Map neuron indices to original Marabou ReLU variables (old indexing)
    std::vector<unsigned> originalVars;
    originalVars.reserve( k );
    for ( unsigned i = 0; i < k; ++i )
    {
        const unsigned var = weightedSumLayer->neuronToVariable( neurons[i] );
        originalVars.push_back( _baseIpqPreprocessor.getOldIndex( var ) );
    }

    ASSERT( originalVars.size() == k );
    ASSERT( forbid.size() == k );

    outDependency = Dependency( originalVars, forbid );
    return true;
}

DependencyAnalyzer::Bitmask DependencyAnalyzer::_buildDependencyBitmask( const std::vector<unsigned> &variables ) const
{
    // Build a bitmask over SAT variable ids (used for minimality checks)
    Bitmask depMask( _bitmaskSize );

    for ( unsigned oldVar : variables )
    {
        const unsigned satVar = reluIndexToSatVar( oldVar );
        if ( satVar == 0 )
        {
            // This path indicates a mismatch between variables we try to encode and the SAT mapping
            ASSERT( false );
        }

        ASSERT( satVar < depMask.size() );
        depMask.set( satVar );
    }

    return depMask;
}

DependencyAnalyzer::Bitmask DependencyAnalyzer::_buildDependencySubBitmask( const std::vector<unsigned> &variables ) const
{
    // Same as _buildDependencyBitmask, but skips unmapped variables (for pruning)
    Bitmask depMask( _bitmaskSize );

    for ( unsigned oldVar : variables )
    {
        const unsigned satVar = reluIndexToSatVar( oldVar );
        if ( satVar == 0 )
            continue;

        ASSERT( satVar < depMask.size() );
        depMask.set( satVar );
    }

    return depMask;
}

bool DependencyAnalyzer::_isNonMinimalDependency( const Bitmask &depMask ) const
{
    // depMask is non-minimal if it is a superset of an existing minimal dependency mask
    for ( const Bitmask &known : _minimalDependencyBitmasks )
    {
        if ( known.size() > depMask.size() )
            continue;

        if ( ( known & depMask ) == known )
            return true;
    }

    return false;
}

void DependencyAnalyzer::_recordMinimalDependencyBitmask( const Bitmask &depMask )
{
    
    _minimalDependencyBitmasks.push_back( depMask );
}

void DependencyAnalyzer::addDependenciesTocadical( const Dependency &dep )
{
    // Encode the dependency as a CNF clause (nogood)
    Vector<int> clauseLits;
    _encodeDependencyToClauseLits( dep, clauseLits );
    ASSERT( !clauseLits.empty() );

    // Push clause into CaDiCaL
    for ( unsigned i = 0; i < clauseLits.size(); ++i )
        _cadical.add( clauseLits[i] );
    _cadical.add( 0 );
}

void DependencyAnalyzer::_encodeDependencyToClauseLits( const Dependency &dep,
                                                       Vector<int> &outClause )
{
    // Each dependency is a forbidden conjunction; encode as disjunction of negated literals
    ASSERT( outClause.empty() );

    const auto &vars   = dep.getVars();
    const auto &states = dep.getStates();

    ASSERT( vars.size() == states.size() );
    ASSERT( !vars.empty() );

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned reluVar = vars[i];
        const ReLUState phase  = states[i];

        const int lit = phaseToLit( reluVar, phase );
        outClause.append( -lit );
    }
}

bool DependencyAnalyzer::recordConflict( Dependency d )
{
    // Record a new dependency and register it with both runtime watchers and SAT.
    ASSERT( d.size() >= 2 );
    ASSERT( d.getVars().size() == d.getStates().size() );

    const std::vector<unsigned> &vars = d.getVars();
    const std::vector<ReLUState> &states = d.getStates();

    // Require canonical ordering
    for ( size_t i = 1; i < vars.size(); ++i )
        ASSERT( vars[i - 1] < vars[i] );

    // Should not already exist (guarded by _dependencyIndex)
    ASSERT( _dependencyIndex.find( d ) == _dependencyIndex.end() );

    // Add to permanent storage
    const DependencyState::DependencyId id = _addDependency( d );
    printf( "[DA] Recorded new dependency id=%u vars=", id );
    for ( unsigned i = 0; i < states.size(); ++i )
    {
        const unsigned v = vars[i];
        const ReLUState s = states[i];
        printf( "%u (%s) ",v , ( s == ReLUState::Active ? "A" : "I" ) );
    }
    printf( "\n" ); 

    // If context exists, also add a backtrackable runtime-state object
    if ( _context )
        _addDependencyRuntimeState( id, d );

    // Add the nogood to CaDiCaL
    addDependenciesTocadical( d );

    // Track minimal masks for pruning
    const Bitmask depMask = _buildDependencyBitmask( vars );
    _recordMinimalDependencyBitmask( depMask );

    return true;
}

void DependencyAnalyzer::_getLayerBounds( const NLR::Layer *layer,
                                         Vector<double> &lowerBounds,
                                         Vector<double> &upperBounds ) const
{
    // Read bounds from the NLR layer and return them as dense vectors
    Vector<double> L, U;
    const unsigned n = layer->getSize();
    for ( unsigned i = 0; i < n; ++i )
    {
        L.append( layer->getLb( i ) );
        U.append( layer->getUb( i ) );
    }
    lowerBounds = L;
    upperBounds = U;
}

void DependencyAnalyzer::_boxMinMax( const Vector<double> &a, double b,
                                     const Vector<double> &L, const Vector<double> &U,
                                     double &outMin, double &outMax ) const
{
    // Compute exact min/max of affine function a·x + b over a box x in [L,U]
    ASSERT( a.size() == L.size() && a.size() == U.size() );
    double mn = b, mx = b;
    for ( unsigned j = 0; j < a.size(); ++j )
    {
        const double aj = a[j];
        if ( aj >= 0 )
        {
            mn += aj * L[j];
            mx += aj * U[j];
        }
        else
        {
            mn += aj * U[j];
            mx += aj * L[j];
        }
    }
    outMin = mn;
    outMax = mx;
}

void DependencyAnalyzer::_sliceMinMax_givenOtherZero( const Vector<double> &w_t, double b_t,
                                                     const Vector<double> &w_o, double b_o,
                                                     const Vector<double> &L, const Vector<double> &U,
                                                     double &outMin, double &outMax ) const
{
    // Approximate min/max of target neuron (w_t·x+b_t) under constraint other neuron is 0:
    //   w_o·x + b_o = 0
    // by eliminating one pivot variable and then bounding over remaining box.
    ASSERT( w_t.size() == w_o.size() && w_t.size() == L.size() && L.size() == U.size() );
    const unsigned dim = w_t.size();
    ASSERT( dim > 0 );

    // Baseline: ignore the equality constraint
    double baseMin = 0.0, baseMax = 0.0;
    _boxMinMax( w_t, b_t, L, U, baseMin, baseMax );

    // Best bounds after trying pivots
    double bestMin = baseMin; // maximize the lower bound
    double bestMax = baseMax; // minimize the upper bound

    // Try each pivot k where w_o[k] != 0 and w_t[k] != 0
    for ( unsigned k = 0; k < dim; ++k )
    {
        const double denom = w_o[k];
        if ( FloatUtils::isZero( denom ) )
            continue;

        if ( FloatUtils::isZero( w_t[k] ) )
            continue;

        // After elimination of x_k, compute transformed constant term
        const double bPrime = b_t - ( w_t[k] * ( b_o / denom ) );

        double mn_k = bPrime;
        double mx_k = bPrime;

        // Bound remaining affine form over the box (excluding x_k)
        for ( unsigned j = 0; j < dim; ++j )
        {
            if ( j == k )
                continue;

            const double coeff = w_t[j] - ( w_t[k] * ( w_o[j] / denom ) );

            if ( coeff >= 0.0 )
            {
                mn_k += coeff * L[j];
                mx_k += coeff * U[j];
            }
            else
            {
                mn_k += coeff * U[j];
                mx_k += coeff * L[j];
            }
        }

        // Update best bounds
        if ( FloatUtils::gt( mn_k, bestMin ) )
            bestMin = mn_k;

        if ( FloatUtils::lt( mx_k, bestMax ) )
            bestMax = mx_k;
    }

    outMin = bestMin;
    outMax = bestMax;
}

bool DependencyAnalyzer::detectAndRecordConflict( unsigned layerIndex,
                                                 const std::vector<unsigned> &neurons )
{
    // Compute a dependency for a given neuron set, prune if non-minimal, and record if found.
    const size_t k = neurons.size();
    ASSERT( k == 2 || k == 3 || k == 4 );

    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( layerIndex );
    ASSERT( weightedSumLayer );

    // Map NLR neuron indices -> original Marabou variables
    std::vector<unsigned> vars;
    vars.reserve( k );
    for ( unsigned n : neurons )
    {
        const unsigned var = weightedSumLayer->neuronToVariable( n );
        const unsigned originalVar = _baseIpqPreprocessor.getOldIndex( var );
        vars.push_back( originalVar );
    }

    // Subset pruning: skip if superset of known minimal dependency
    const Bitmask depMask = _buildDependencySubBitmask( vars );
    bool issuperset = _isNonMinimalDependency( depMask );

    if ( issuperset )
        return false;

    Dependency d;
    const bool found = analyzeConflict( layerIndex, neurons, d );
    if ( !found )
        return false;

    return recordConflict( std::move( d ) );
}

#ifdef ENABLE_GUROBI
void DependencyAnalyzer::_buildLpSliceModelMEq( GurobiWrapper &lp,
                                               const Vector<Vector<double>> &w_eq,
                                               const Vector<double> &b_eq,
                                               const Vector<double> &L,
                                               const Vector<double> &U,
                                               Vector<String> &varNames ) const
{
    // Build LP:
    //   variables z_j in [L_j, U_j]
    //   constraints: w_eq[i]·z + b_eq[i] = 0   (rewritten in wrapper form)
    const unsigned dim = L.size();
    const unsigned m   = w_eq.size();
    ASSERT( m == b_eq.size() );

    lp.resetModel();

    varNames = Vector<String>( dim );
    for ( unsigned j = 0; j < dim; ++j )
    {
        const String name = Stringf( "z_%u", j );
        varNames[j] = name;
        lp.addVariable( name, L[j], U[j], GurobiWrapper::CONTINUOUS );
    }

    for ( unsigned i = 0; i < m; ++i )
    {
        List<GurobiWrapper::Term> eqTerms;
        for ( unsigned j = 0; j < dim; ++j )
            if ( !FloatUtils::isZero( w_eq[i][j] ) )
                eqTerms.append( GurobiWrapper::Term( w_eq[i][j], varNames[j] ) );

        // Wrapper expects sum terms == -b
        lp.addEqConstraint( eqTerms, -b_eq[i] );
    }
}
#endif

#ifdef ENABLE_GUROBI
bool DependencyAnalyzer::lpSliceMEqMinMax( const Vector<double> &w_t, double b_t,
                                          const Vector<Vector<double>> &w_eq,
                                          const Vector<double> &b_eq,
                                          const Vector<double> &L,
                                          const Vector<double> &U,
                                          double &outMin, double &outMax ) const
{
    // Solve two LPs: minimize and maximize w_t·z + b_t under the equality slice constraints.
    ASSERT( w_t.size() == L.size() && L.size() == U.size() );

    GurobiWrapper &lp = _lpReusable;

    // One-time LP configuration (quiet, single-thread, small time limit)
    if ( !_lpReusableInitialized )
    {
        lp.setNumberOfThreads( 1 );
        lp.setVerbosity( 0 );
        lp.setTimeLimit( 0.05 );
        _lpReusableInitialized = true;
    }

    Vector<String> varNames;
    _buildLpSliceModelMEq( lp, w_eq, b_eq, L, U, varNames );

    // Build objective terms once
    List<GurobiWrapper::Term> objTerms;
    for ( unsigned j = 0; j < w_t.size(); ++j )
        if ( !FloatUtils::isZero( w_t[j] ) )
            objTerms.append( GurobiWrapper::Term( w_t[j], varNames[j] ) );

    bool isOk = true;

    // MIN
    lp.setCost( objTerms, b_t );
    lp.solve();
    if ( lp.infeasible() || !lp.haveFeasibleSolution() )
    {
        outMin = FloatUtils::infinity();
        isOk = false;
    }
    else
        outMin = lp.getOptimalCostOrObjective();

    // MAX (same constraints, new objective direction)
    lp.setObjective( objTerms, b_t );
    lp.solve();
    if ( lp.infeasible() || !lp.haveFeasibleSolution() )
    {
        outMax = -FloatUtils::infinity();
        isOk = false;
    }
    else
        outMax = lp.getOptimalCostOrObjective();

    return isOk;
}
#endif

#ifdef ENABLE_GUROBI
void DependencyAnalyzer::_sliceMinMax_givenMEqZero_LP( const Vector<double> &w_t, double b_t,
                                                      const Vector<Vector<double>> &w_eq,
                                                      const Vector<double> &b_eq,
                                                      const Vector<double> &L,
                                                      const Vector<double> &U,
                                                      double &outMin, double &outMax ) const
{
    // Wrapper helper that fills outMin/outMax; on infeasible slice, returns +/- infinity
    double mn = 0.0, mx = 0.0;
    (void)lpSliceMEqMinMax( w_t, b_t, w_eq, b_eq, L, U, mn, mx );
    outMin = mn;
    outMax = mx;
}
#endif

bool DependencyAnalyzer::notifyNeuronFixed( unsigned newVar, ReLUState state )
{
    return notifyNeuronFixed( newVar, state, true );
}

void DependencyAnalyzer::notifyLowerBoundUpdate( unsigned newVar, double oldLb, double newLb )
{
    return notifyLowerBoundUpdate( newVar, oldLb, newLb, true );
}

void DependencyAnalyzer::notifyUpperBoundUpdate( unsigned newVar, double oldUb, double newUb )
{
    return notifyUpperBoundUpdate( newVar, oldUb, newUb, true );
}

bool DependencyAnalyzer::notifyNeuronFixed( unsigned newVar, ReLUState state, bool countForScore)
{
    // Called by Engine when a pre-activation bound crosses 0 (or hits exactly 0 edge cases).
    ASSERT( _seenPhase );
    ASSERT( _preprocessor );

    // Map Engine var (new) -> old var id
    const unsigned var = _preprocessor->getOldIndex( newVar );

    // Map old var -> analyzer-preprocessed var id
    const unsigned newBQVar = _baseIpqPreprocessor.getNewIndex( var );

    // Read current analyzer bounds for this var
    const double lb = _preprocessedQuery->getLowerBound( newBQVar );
    const double ub = _preprocessedQuery->getUpperBound( newBQVar );

    // Allow:
    //  (1) originally unstable vars, OR
    //  (2) stable vars that become exactly at zero boundary
    const bool originallyUnstable = _isUnstableVar( var );
    const bool allowed =
        originallyUnstable ||
        ( state == ReLUState::Inactive && FloatUtils::isZero( lb ) ) ||
        ( state == ReLUState::Active   && FloatUtils::isZero( ub ) );

    // Currently informational: we do not block execution; keep it available for future asserts
    (void)allowed; // TODO: assert allowed;
    (void)lb;
    (void)ub;

    // Convert to runtime-state enum
    ReLURuntimeState incoming =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Active : ReLURuntimeState::Inactive;

    const ReLURuntimeState opposite =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Inactive : ReLURuntimeState::Active;

    // If both phases were seen across time, treat as Zero (for now)
    auto ait = _seenPhase->find( var );
    if ( ait != _seenPhase->end() && ( *ait ).second == opposite )
        incoming = ReLURuntimeState::Zero;

    // If present, must be the opposite (otherwise this is inconsistent)
    ASSERT( ait == _seenPhase->end() || ( *ait ).second == opposite );

    _seenPhase->insert( var, incoming );

    // Bump score for unstable vars
    if ( countForScore && _isUnstableVar( var ) )
    {
        _markFixedNow( var );
        _bumpScore( var );
    }
    else
    {
        // Stable vars do not contribute to score
        // printf( "[DA][score] skipping stable var %u for score bump\n", var );
    }

    return true;
}

void DependencyAnalyzer::notifyLowerBoundUpdate( unsigned newVar,
                                                double previousLowerBound,
                                                double newLowerBound,
                                                bool countForScore)
{
    // Lower bounds must only tighten upward
    ASSERT( _preprocessor );

    if ( !FloatUtils::gt( newLowerBound, previousLowerBound ) )
        return;
    ASSERT( !FloatUtils::lt( newLowerBound, previousLowerBound ) );

    // Crossed 0 from below => neuron is guaranteed Active
    if ( previousLowerBound < 0.0 && newLowerBound >= 0.0 )
        notifyNeuronFixed( newVar, ReLUState::Active, countForScore);
}

void DependencyAnalyzer::notifyUpperBoundUpdate( unsigned newVar,
                                                double previousUpperBound,
                                                double newUpperBound,
                                                bool countForScore )
{
    // Upper bounds must only tighten downward
    ASSERT( _preprocessor );

    if ( !FloatUtils::lt( newUpperBound, previousUpperBound ) )
        return;
    ASSERT( !FloatUtils::gt( newUpperBound, previousUpperBound ) );

    // Crossed 0 from above => neuron is guaranteed Inactive
    if ( previousUpperBound > 0.0 && newUpperBound <= 0.0 )
        notifyNeuronFixed( newVar, ReLUState::Inactive, countForScore);
}

DependencyState::DependencyId DependencyAnalyzer::_addDependency( const Dependency &d )
{
    // Append dependency and remember its id for future duplicate checks
    const DependencyState::DependencyId id = _dependencies.size();
    _dependencies.push_back( d );
    _dependencyIndex.emplace( d, id );
    return id;
}

void DependencyAnalyzer::_addDependencyRuntimeState( DependencyState::DependencyId id,
                                                    const Dependency &d )
{
    // Runtime states are context-dependent (backtrackable)
    ASSERT( _context );

    DependencyState st( id, static_cast<unsigned>( d.size() ), *_context );
    _dependencyStates.push_back( std::move( st ) );
}

void DependencyAnalyzer::_computeCoveringBoxFromRemainingQueries()
{
    // Compute the smallest box that contains all remaining query boxes
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

    // --------------------
    // Stats on box widths
    // --------------------
    double minDiff = +INFINITY;
    double maxDiff = -INFINITY;
    double sumDiff = 0.0;

    std::vector<double> diffs;
    diffs.reserve( _inputDim );

    for ( unsigned x = 0; x < _inputDim; ++x )
    {
        const double diff = _currentUb[x] - _currentLb[x];

        // If you want to sanity-check:
        // ASSERT( diff >= 0.0 );

        diffs.push_back( diff );
        sumDiff += diff;
        minDiff = std::min( minDiff, diff );
        maxDiff = std::max( maxDiff, diff );
    }

    const double meanDiff = sumDiff / _inputDim;

    // Median
    std::sort( diffs.begin(), diffs.end() );
    double medianDiff;
    if ( _inputDim % 2 == 0 )
    {
        medianDiff = 0.5 * ( diffs[_inputDim / 2 - 1] +
                            diffs[_inputDim / 2] );
    }
    else
    {
        medianDiff = diffs[_inputDim / 2];
    }

    // Standard deviation
    double var = 0.0;
    for ( double d : diffs )
        var += ( d - meanDiff ) * ( d - meanDiff );
    var /= _inputDim;

    const double stdDiff = std::sqrt( var );

    // Print
    printf(
        "[DA] CoveringBox stats (dims=%u): "
        "diff[min=%.6g, max=%.6g, mean=%.6g, median=%.6g, std=%.6g]\n",
        _inputDim,
        minDiff,
        maxDiff,
        meanDiff,
        medianDiff,
        stdDiff
    );

}

bool DependencyAnalyzer::_isSubset( const Vector<double> &lbNew,
                                   const Vector<double> &ubNew,
                                   const Vector<double> &lbOld,
                                   const Vector<double> &ubOld ) const
{
    // Check if [lbNew,ubNew] ⊆ [lbOld,ubOld] componentwise
    for ( unsigned x = 0; x < _inputDim; ++x )
    {
        if ( lbNew[x] < lbOld[x] ) return false;
        if ( ubNew[x] > ubOld[x] ) return false;
    }
    return true;
}

void DependencyAnalyzer::notifyQuerySolved()
{
    // Called after a query in the batch is solved; shrink the covering box over remaining queries.
    ASSERT( _nextQueryToSolve < _numQueries );
    ++_nextQueryToSolve;

    if ( _nextQueryToSolve >= _numQueries )
        return;

    const Vector<double> oldLb = _currentLb;
    const Vector<double> oldUb = _currentUb;

    _computeCoveringBoxFromRemainingQueries();

    // Build tightenings for input variables based on new covering box
    List<Tightening> inputTightenings;

    // Layer 0 is assumed to correspond to inputs in this setup
    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( 0 );
    if ( !weightedSumLayer )
        return;

    const unsigned numNeurons = weightedSumLayer->getSize();
    ASSERT( _inputDim == numNeurons );

    for ( unsigned i = 0; i < _inputDim; ++i )
    {
        // Map input index i -> variable in the analyzer preprocessed query
        const unsigned newVar = weightedSumLayer->neuronToVariable( i );

        // Inputs are assumed never eliminated; oldVar == newVar in this code path
        const unsigned oldVar = _baseIpqPreprocessor.getOldIndex( newVar );
        ASSERT( newVar == oldVar );

        const unsigned var = oldVar;

        const double oldL = oldLb[i];
        const double oldU = oldUb[i];
        const double newL = _currentLb[i];
        const double newU = _currentUb[i];

        if ( FloatUtils::gt( newL, oldL ) )
            inputTightenings.append( Tightening( var, newL, Tightening::LB ) );

        if ( FloatUtils::lt( newU, oldU ) )
            inputTightenings.append( Tightening( var, newU, Tightening::UB ) );
    }

    // Apply any input tightenings to the analyzer query
    if ( !inputTightenings.empty() )
        _applyTighteningsToPreprocessedQuery( inputTightenings );

    // New covering box must be a subset of the old one
    ASSERT( _isSubset( _currentLb, _currentUb, oldLb, oldUb ) );

    // Clear per-query/context state; Engine will setContext again for next query
    _context = nullptr;
    _preprocessor = nullptr;
    _seenPhase = nullptr;
    _dependencyStates.clear();
}

void DependencyAnalyzer::_collectAllUnstableNeurons()
{
    // Collect all (old-index) ReLU variables whose pre-activation bounds cross 0.
    _unstableNeurons.clear();

    if ( !_networkLevelReasoner )
        return;

    // Ensure NLR bounds match the analyzer's preprocessed query bounds
    _networkLevelReasoner->obtainCurrentBounds( *_preprocessedQuery );

    const unsigned numLayers = _networkLevelReasoner->getNumberOfLayers();

    for ( unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex )
    {
        const NLR::Layer *layer = _networkLevelReasoner->getLayer( layerIndex );
        if ( !layer )
            continue;

        if ( layer->getLayerType() == NLR::Layer::WEIGHTED_SUM )
        {
            std::vector<unsigned> unstableIndices;
            collectUnstableNeurons( layerIndex, false, unstableIndices);

            // Convert neuron indices to old-variable ids
            for ( unsigned neuronIndex : unstableIndices )
            {
                const unsigned newVar = layer->neuronToVariable( neuronIndex );
                const unsigned var = _baseIpqPreprocessor.getOldIndex( newVar );
                _unstableNeurons.push_back( var );
            }
        }
    }

    // Keep sorted unique list (needed for binary_search)
    std::sort( _unstableNeurons.begin(), _unstableNeurons.end() );
    _unstableNeurons.erase( std::unique( _unstableNeurons.begin(), _unstableNeurons.end() ),
                            _unstableNeurons.end() );
}

bool DependencyAnalyzer::_isUnstableVar( unsigned var ) const
{
    // Membership query on sorted unstable list
    return std::binary_search( _unstableNeurons.begin(), _unstableNeurons.end(), var );
}

void DependencyAnalyzer::syncWithEnginePreprocessedQuery( const Query &engineQuery )
{
    // Sync already-tightened bounds from Engine and emit fixed-phase notifications.
    ASSERT( _preprocessedQuery );
    ASSERT( _networkLevelReasoner );
    ASSERT( _context );
    ASSERT( _preprocessor );

    if ( _unstableNeurons.empty() )
        return;

    for ( unsigned oldVar : _unstableNeurons )
    {
        // Convert old -> engine new index
        const unsigned var = _preprocessor->getNewIndex( oldVar );

        const double lb = engineQuery.getLowerBound( var );
        const double ub = engineQuery.getUpperBound( var );

        // If already guaranteed Active / Inactive, notify using a sentinel "previous" bound
        if ( !FloatUtils::lt( lb, 0.0 ) )
            notifyLowerBoundUpdate( var, -INFINITY, lb, false );
        else if ( !FloatUtils::gt( ub, 0.0 ) )
            notifyUpperBoundUpdate( var, +INFINITY, ub, false );
    }
}

void DependencyAnalyzer::_initializeSatSolver()
{
    // Reserve index 0 so SAT vars 1..N map naturally to _satVarToReluIndex[1..N]
    _satVarToReluIndex.append( (unsigned)-1 );
}

ReLURuntimeState DependencyAnalyzer::_getReluPhase( unsigned reluVar ) const
{
    // Return current runtime phase from _seenPhase map; Zero treated as Unstable for SAT purposes
    ASSERT( _seenPhase );

    auto it = _seenPhase->find( reluVar );
    if ( it == _seenPhase->end() )
        return ReLURuntimeState::Unstable;

    const ReLURuntimeState state = ( *it ).second;

    if ( state == ReLURuntimeState::Zero )
        return ReLURuntimeState::Unstable;

    return state;
}

unsigned DependencyAnalyzer::reluIndexToSatVar( unsigned reluVar ) const
{
    // Query existing mapping; 0 means "not mapped"
    auto it = _reluIndexToSatVar.find( reluVar );
    if ( it == _reluIndexToSatVar.end() )
        return 0;
    return it->second;
}

unsigned DependencyAnalyzer::reluIndexToSatVarForce( unsigned reluVar )
{
    // Create mapping on demand if missing
    const unsigned existing = reluIndexToSatVar( reluVar );
    if ( existing != 0 )
        return existing;

    return _createNewSatVarForRelu( reluVar );
}

unsigned DependencyAnalyzer::_createNewSatVarForRelu( unsigned reluVar )
{
    // Assign next SAT variable id and record bi-directional maps
    const unsigned newSatVar = _satVarToReluIndex.size(); // SAT vars start at 1
    ASSERT( newSatVar > 0 );

    _reluIndexToSatVar[reluVar] = newSatVar;
    _satVarToReluIndex.append( reluVar );

    return newSatVar;
}

unsigned DependencyAnalyzer::satVarToReluIndex( unsigned satVar ) const
{
    // SAT variables are 1-based in our mapping
    ASSERT( satVar > 0 );
    ASSERT( satVar < _satVarToReluIndex.size() );

    return _satVarToReluIndex[satVar];
}

int DependencyAnalyzer::phaseToLit( unsigned reluVar, ReLUState phase )
{
    // Encode phase as a SAT literal:
    //   Active   ->  +satVar
    //   Inactive ->  -satVar
    ASSERT( phase == ReLUState::Active || phase == ReLUState::Inactive );

    const unsigned satVar = reluIndexToSatVarForce( reluVar );
    return ( phase == ReLUState::Active ) ? (int)satVar : -(int)satVar;
}

bool DependencyAnalyzer::litToPhase( int lit, unsigned &reluVar, ReLUState &phase )
{
    // Decode a SAT literal into (reluVar, phase)
    ASSERT( lit != 0 );

    const unsigned satVar = ( lit > 0 ) ? (unsigned)lit : (unsigned)( -lit );
    ASSERT( satVar < _satVarToReluIndex.size() );

    reluVar = satVarToReluIndex( satVar );
    phase   = ( lit > 0 ) ? ReLUState::Active : ReLUState::Inactive;

    return true;
}

void DependencyAnalyzer::_emitTighteningsForImpliedPhase( unsigned reluVar,
                                                         ReLUState impliedPhase,
                                                         List<Tightening> &tightenings )
{
    // Convert a SAT-implied phase into an Engine tightening on the corresponding variable.
    ASSERT( _preprocessedQuery );

    // Analyzer preprocessed var id
    const unsigned reluVarBase = _baseIpqPreprocessor.getNewIndex( reluVar );
    const double lb = _preprocessedQuery->getLowerBound( reluVarBase );
    const double ub = _preprocessedQuery->getUpperBound( reluVarBase );

    if ( impliedPhase == ReLUState::Active )
    {
        // Active => pre-activation >= 0
        const double newLb = 0.0;

        // Safety: cannot exceed current UB; must strengthen
        ASSERT( !FloatUtils::gt( newLb, ub ) );
        ASSERT( FloatUtils::gt( newLb, lb ) );

        // Engine var id
        const unsigned engineVar = _preprocessor->getNewIndex( reluVar );
        tightenings.append( Tightening( engineVar, newLb, Tightening::LB ) );
    }
    else
    {
        // Inactive => pre-activation <= 0
        ASSERT( impliedPhase == ReLUState::Inactive );
        const double newUb = 0.0;

        // Safety: cannot go below current LB; must strengthen
        ASSERT( !FloatUtils::lt( newUb, lb ) );
        ASSERT( FloatUtils::lt( newUb, ub ) );

        // Engine var id
        const unsigned engineVar = _preprocessor->getNewIndex( reluVar );
        tightenings.append( Tightening( engineVar, newUb, Tightening::UB ) );
    }
}

void DependencyAnalyzer::getImpliedTighteningsFromSat( List<Tightening> &tightenings, bool calculateDependencies )
{
    if ( calculateDependencies )
    {
            computeSameLayerDependencies();
    }

    // Propagate current assumptions in CaDiCaL and convert entailed literals to tightenings.
    // (CNF dumping is disabled inside debugPrintSatClauses()).
    debugPrintSatClauses();

    // Add assumptions for all currently fixed ReLUs in our SAT mapping
    for ( const auto &entry : _reluIndexToSatVar )
    {
        const unsigned reluVar = entry.first;
        const ReLURuntimeState rt = _getReluPhase( reluVar );

        if ( rt == ReLURuntimeState::Active )
            _cadical.assume( phaseToLit( reluVar, ReLUState::Active ) );
        else if ( rt == ReLURuntimeState::Inactive )
            _cadical.assume( phaseToLit( reluVar, ReLUState::Inactive ) );
        // Zero/Unstable: no assumption
    }

    // Propagate under assumptions
    const int res = _cadical.propagate();

    // 20 means conflict under assumptions; no implications to emit
    if ( res == 20 )
        return;

    // Query entailed literals (implications)
    std::vector<int> implicants;
    _cadical.get_entrailed_literals( implicants );

    for ( int lit : implicants )
    {
        unsigned reluVar = 0;
        ReLUState impliedPhase;

        const bool success = litToPhase( lit, reluVar, impliedPhase );
        ASSERT( success );

        const ReLURuntimeState currentRt = _getReluPhase( reluVar );

        // Skip if already fixed consistently
        if ( ( currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Active ) ||
             ( currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Inactive ) )
        {
            continue;
        }

        // Contradiction: SAT implies opposite of runtime-fixed value
        if ( ( currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Inactive ) ||
             ( currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Active ) )
        {
            ASSERT( false );
        }

        // Zero is treated as neither active nor inactive; do not allow further forcing
        if ( currentRt == ReLURuntimeState::Zero )
        {
            ASSERT( false );
        }

        // Emit tightening for newly implied phase
        _emitTighteningsForImpliedPhase( reluVar, impliedPhase, tightenings );
    }
}

void DependencyAnalyzer::_markFixedNow( unsigned oldVar )
{
    ++_fixCounter;
    _unstableNeuronLastFixed[oldVar] = _fixCounter;
}

void DependencyAnalyzer::_bumpScore( unsigned oldVar )
{
    // simplest: +1 per fix event
    _unstableNeuronScores[oldVar] += 1.0;
    // printf("[Debug][score] Bumping score for unstable var %u to %.4f\n", oldVar, _unstableNeuronScores[oldVar]);
}


double DependencyAnalyzer::_getScore( unsigned oldVar ) const
{
    auto it = _unstableNeuronScores.find( oldVar );
    return it == _unstableNeuronScores.end() ? 0.0 : it->second;
}

void DependencyAnalyzer::_pruneUnstableByTopKWithRecency( unsigned weightedSumLayerIndex,
                                                         std::vector<unsigned> &unstable,
                                                         double fractionToKeep,
                                                         unsigned minK,
                                                         unsigned maxK ) const
{
    const NLR::Layer *weightedSumLayer = _networkLevelReasoner->getLayer( weightedSumLayerIndex );
    if ( !weightedSumLayer )
        return;

    // Build candidates: only those with score > 0
    struct Cand
    {
        unsigned neuronIndex;
        unsigned oldVar;
        double score;
        uint64_t lastFixed;
    };

    // Reserve pessimistically (same as unstable size)
    std::vector<Cand> cands;
    cands.reserve( unstable.size() );

    for ( unsigned neuronIndex : unstable )
    {
        const unsigned var    = weightedSumLayer->neuronToVariable( neuronIndex );
        const unsigned oldVar = _baseIpqPreprocessor.getOldIndex( var );

        auto sit = _unstableNeuronScores.find( oldVar );
        if ( sit == _unstableNeuronScores.end() )
            continue;

        const double s = sit->second;
        if ( s <= 0.0 )
            continue;

        uint64_t lf = 0;
        auto lit = _unstableNeuronLastFixed.find( oldVar );
        if ( lit != _unstableNeuronLastFixed.end() )
            lf = lit->second;

        cands.push_back( { neuronIndex, oldVar, s, lf } );
    }

    // If nobody has score>0 yet => prune everything (so no deps initially)
    if ( cands.empty() )
    {
        unstable.clear();
        return;
    }

    // Decide K based on scored count
    unsigned K = (unsigned)std::ceil( fractionToKeep * (double)cands.size() );
    if ( K < minK ) K = minK;
    if ( K > maxK ) K = maxK;
    if ( K > cands.size() ) K = cands.size();

    if ( cands.size() <= minK )
    {
        // Keep all scored candidates (still need to rewrite `unstable`)
        unstable.clear();
        unstable.reserve( cands.size() );
        for ( const auto &c : cands )
            unstable.push_back( c.neuronIndex );
        return;
    }

    // Comparator: higher score first; tie -> more recent; tie -> smaller oldVar (deterministic)
    auto better = []( const Cand &a, const Cand &b ) {
        if ( a.score != b.score ) return a.score > b.score;
        if ( a.lastFixed != b.lastFixed ) return a.lastFixed > b.lastFixed;
        return a.oldVar < b.oldVar;
    };

    // Partition so that first K are the best K (O(n))
    std::nth_element( cands.begin(), cands.begin() + K, cands.end(), better );
    cands.resize( K );

    // Optional: make selection order deterministic (sort only K items, cheap)
    std::sort( cands.begin(), cands.end(), better );

    unstable.clear();
    unstable.reserve( K );
    for ( const auto &c : cands )
        unstable.push_back( c.neuronIndex );
}

void DependencyAnalyzer::setCurrentNetworkLevelReasoner( NLR::NetworkLevelReasoner *nlr )
{
    _currNetworkLevelReasoner = nlr;
}



void DependencyAnalyzer::debugPrintSatClauses()
{
    // Disabled: originally dumped CNF via write_dimacs + file readback
    return;
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
