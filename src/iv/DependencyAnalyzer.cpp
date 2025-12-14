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
#include <chrono>
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
    _preprocessor = nullptr;
    _seenPhase = nullptr;

    buildFromBase();
    std::printf("[DA] initial _baseIpq: vars=%u, eqs=%u\n",
        _baseIpq->getNumberOfVariables(),
        _baseIpq->getNumberOfEquations());

    _collectAllUnstableNeurons();
    _bitmaskSize = _unstableNeurons.size();
    _initializeSatSolver();
}

void DependencyAnalyzer::buildFromBase()
{
    if ( !_baseIpq )
    {
        printf("DependencyAnalyzer::buildFromBase called with null baseIpq." );
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
            "DependencyAnalyzer::buildFromBase called with null baseIpq." );
            
    }

    // preprocess returns std::unique_ptr<Query>; assign/move it directly
    _preprocessedQuery = _baseIpqPreprocessor.preprocess(
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
void DependencyAnalyzer::setPreprocessor( Preprocessor *preprocessor )
{
    _preprocessor = preprocessor;
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
        unsigned var = weightedSumLayer->neuronToVariable( neuronIndex );
        unsigned originalVar = _baseIpqPreprocessor.getOldIndex( var );

        double nlrLb = weightedSumLayer->getLb( neuronIndex );
        double pqLb  = _preprocessedQuery->getLowerBound( var );

        double nlrUb = weightedSumLayer->getUb( neuronIndex );
        double pqUb  = _preprocessedQuery->getUpperBound( var );

        if ( !FloatUtils::areEqual( nlrLb, pqLb, 1e-9 ) ||
            !FloatUtils::areEqual( nlrUb, pqUb, 1e-9 ) )
        {
            printf("[DA][warn] bound mismatch at layer %u neuron %u "
                "(var %u, old var %u): NLR [%.6g, %.6g] vs PQ [%.6g, %.6g]\n",
                layerIndex, neuronIndex, var, originalVar, nlrLb, nlrUb, pqLb, pqUb);
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
    unsigned addedPairs   = 0;
    unsigned addedTriples = 0;
    unsigned addedQuads = 0;

    // enumerate unordered pairs
    for ( size_t i = 0; i + 1 < unstable.size(); ++i )
    {
        unsigned q = unstable[i];
        for ( size_t j = i + 1; j < unstable.size(); ++j )
        {
            unsigned r = unstable[j];
            ASSERT( q < r );
            if ( detectAndRecordConflict( weightedSumLayerIndex, { q, r } ) )
                ++addedPairs;
        }
    }

    // enumerate unordered triples
    if ( unstable.size() >= 3 )
    {
        for ( size_t i = 0; i + 2 < unstable.size(); ++i )
        {
            unsigned q = unstable[i];
            for ( size_t j = i + 1; j + 1 < unstable.size(); ++j )
            {
                unsigned r = unstable[j];
                for ( size_t k = j + 1; k < unstable.size(); ++k )
                {
                    unsigned s = unstable[k];
                    ASSERT( q < r && r < s );
                    if ( detectAndRecordConflict( weightedSumLayerIndex, { q, r, s } ) )
                        ++addedTriples;
                }
            }
        }
    }

    // enumerate unordered quadruples
    if ( unstable.size() >= 4 )
    {
        for ( size_t i = 0; i + 3 < unstable.size(); ++i )
        {
            unsigned q = unstable[i];
            for ( size_t j = i + 1; j + 2 < unstable.size(); ++j )
            {
                unsigned r = unstable[j];
                for ( size_t k = j + 1; k + 1 < unstable.size(); ++k )
                {
                    unsigned s = unstable[k];
                    for ( size_t m = k + 1; m < unstable.size(); ++m )
                    {
                        unsigned t = unstable[m];
                        ASSERT( q < r && r < s && s < t );
                        if ( detectAndRecordConflict( weightedSumLayerIndex,  { q, r, s, t } ) )
                            ++addedQuads;
                    }
                }
            }
        }
    }


    const size_t scannedPairs =
        ( unstable.size() * ( unstable.size() - 1 ) ) / 2;

    const size_t scannedTriples =
        ( unstable.size() >= 3 ) ?
        ( unstable.size() * ( unstable.size() - 1 ) * ( unstable.size() - 2 ) ) / 6 :
        0;
    const size_t scannedQuads =
        ( unstable.size() >= 4 ) ?
        ( unstable.size() * ( unstable.size() - 1 ) * ( unstable.size() - 2 ) * ( unstable.size() - 3 ) ) / 24 :
        0;

    printf( "[DA] layer %u: scanned %zu pairs, added %u pair-conflicts; "
            "scanned %zu triples, added %u triple-conflicts, "
            "scanned %zu quads, added %u quad-conflicts.\n",
            weightedSumLayerIndex,
            scannedPairs, addedPairs,
            scannedTriples, addedTriples,
            scannedQuads, addedQuads );

    return addedPairs + addedTriples + addedQuads;
}

bool DependencyAnalyzer::analyzeConflict( unsigned layerIndex,
                                         const std::vector<unsigned> &neurons,
                                         Dependency &outDependency )
{
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
        if ( i > 0 ) ASSERT( neurons[i-1] < neurons[i] );
    }
    (void)layerSize;

    const auto &sources = weightedSumLayer->getSourceLayers();
    ASSERT( sources.size() == 1 );
    const unsigned prevLayerIndex = sources.begin()->first;
    const NLR::Layer *prevLayer = _networkLevelReasoner->getLayer( prevLayerIndex );
    ASSERT( prevLayer );
    const unsigned prevSize = prevLayer->getSize();

    // Gather prev-layer bounds
    Vector<double> lowerPrev, upperPrev;
    _getLayerBounds( prevLayer, lowerPrev, upperPrev );

    // Collect weights/biases for each neuron in this conflict
    std::vector<Vector<double>> W;
    std::vector<double> B;
    W.reserve( k );
    B.reserve( k );

    for ( unsigned idx = 0; idx < k; ++idx )
    {
        unsigned n = neurons[idx];
        Vector<double> w( prevSize );
        for ( unsigned j = 0; j < prevSize; ++j )
            w[j] = weightedSumLayer->getWeight( prevLayerIndex, j, n );

        W.push_back( w );
        B.push_back( weightedSumLayer->getBias( n ) );
    }

    // For each i: compute bounds for neuron i under "others = 0"
    std::vector<double> Lcond( k, 0.0 ), Ucond( k, 0.0 );

    for ( unsigned i = 0; i < k; ++i )
    {
        if ( k == 2 )
        {
            // Analytic only for size-2
            unsigned o = ( i == 0 ? 1 : 0 );
            _sliceMinMax_givenOtherZero( W[i], B[i], W[o], B[o],
                                         lowerPrev, upperPrev,
                                         Lcond[i], Ucond[i] );
        }
        else
        {
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

    // If any slice is infeasible, ignore
    for ( unsigned i = 0; i < k; ++i )
    {
        if ( !FloatUtils::isFinite( Lcond[i] ) || !FloatUtils::isFinite( Ucond[i] ) )
            return false;
    }

    // Determine forced phases and require all forced (same rule you use for triple/quad)
    std::vector<bool> forcedActive( k, false ), forcedInactive( k, false );
    for ( unsigned i = 0; i < k; ++i )
    {
        forcedActive[i]   = FloatUtils::gt( Lcond[i], 0.0 );
        forcedInactive[i] = FloatUtils::lt( Ucond[i], 0.0 );
    }

    auto isForced = []( bool fa, bool fi ) {
        return ( fa && !fi ) || ( fi && !fa );
    };

    for ( unsigned i = 0; i < k; ++i )
        if ( !isForced( forcedActive[i], forcedInactive[i] ) )
            return false;

    // Build forbidden assignment: forced_inactive => forbid Active, forced_active => forbid Inactive
    std::vector<ReLUState> forbid( k, ReLUState::Inactive );
    for ( unsigned i = 0; i < k; ++i )
        forbid[i] = forcedInactive[i] ? ReLUState::Active : ReLUState::Inactive;

    // Map neuron indices -> original Marabou variables
    std::vector<unsigned> originalVars;
    originalVars.reserve( k );
    for ( unsigned i = 0; i < k; ++i )
    {
        unsigned var = weightedSumLayer->neuronToVariable( neurons[i] );
        originalVars.push_back( _baseIpqPreprocessor.getOldIndex( var ) );
    }

    // Emit Dependency object
    ASSERT( originalVars.size() == k );
    ASSERT( forbid.size() == k );

    outDependency = Dependency( originalVars, forbid );
    return true;
}

DependencyAnalyzer::Bitmask DependencyAnalyzer::_buildDependencyBitmask(const std::vector<unsigned> &variables) const
{
    Bitmask depMask(_bitmaskSize);  // assumes _bitmaskSize = _preprocessedQuery->getNumberOfVariables()

    for (unsigned oldVar : variables)
    {
        unsigned satVar = reluIndexToSatVar(oldVar);
        if (satVar == 0)
        {
            printf("[DA][bitmask] ERROR: reluVar %u has no SAT var mapping\n", oldVar);
            ASSERT(false);
        }

        ASSERT(satVar < depMask.size());
        depMask.set(satVar);
    }

    return depMask;
}
DependencyAnalyzer::Bitmask DependencyAnalyzer::_buildDependencySubBitmask(const std::vector<unsigned> &variables) const
{
    Bitmask depMask(_bitmaskSize);  // assumes _bitmaskSize = _preprocessedQuery->getNumberOfVariables()

    for (unsigned oldVar : variables)
    {
        unsigned satVar = reluIndexToSatVar(oldVar);
        if (satVar == 0)
        {
            continue;  // skip variables with no SAT mapping
        }

        ASSERT(satVar < depMask.size());
        depMask.set(satVar);
    }

    return depMask;
}

bool DependencyAnalyzer::_isNonMinimalDependency(const Bitmask &depMask) const
{
    for (const Bitmask &known : _minimalDependencyBitmasks)
    {
        if (known.size() > depMask.size())
            continue;

        if ((known & depMask) == known)
            return true;  // depMask is a superset of an existing minimal dep
    }

    return false;
}

void DependencyAnalyzer::_recordMinimalDependencyBitmask(const Bitmask &depMask)
{
    // Remove supersets of depMask
    std::vector<size_t> toRemove;
    for (size_t i = 0; i < _minimalDependencyBitmasks.size(); ++i)
    {
        const Bitmask &existing = _minimalDependencyBitmasks[i];
        Bitmask resized = depMask;
        resized.resize(existing.size());

        if ((existing & resized) == resized)
            toRemove.push_back(i);
    }

    for (auto it = toRemove.rbegin(); it != toRemove.rend(); ++it)
        _minimalDependencyBitmasks.erase(_minimalDependencyBitmasks.begin() + *it);

    _minimalDependencyBitmasks.push_back(depMask);
    printf("[DA][bitmask] inserted new minimal dep mask, size=%zu\n", depMask.count());
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


void DependencyAnalyzer::addDependenciesTocadical( const Dependency &dep )
{
    printf("\n[DA][SAT] ================================================\n");
    printf("[DA][SAT] Adding dependency to CaDiCaL\n");

    // Pretty-print the dependency
    const auto &vars   = dep.getVars();
    const auto &states = dep.getStates();

    printf("[DA][SAT]   Dependency (nogood): ");
    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const char *phaseStr =
            (states[i] == ReLUState::Active)   ? "A" :
            (states[i] == ReLUState::Inactive) ? "I" :
                                                 "?";
        printf("(%u,%s) ", vars[i], phaseStr);
    }
    printf("\n");

    // Encode the nogood clause
    Vector<int> clauseLits;
    _encodeDependencyToClauseLits( dep, clauseLits );

    ASSERT( !clauseLits.empty() );

    // Print encoded clause (before sending to CaDiCaL)
    printf("[DA][SAT]   Encoded clause: ");
    for ( unsigned i = 0; i < clauseLits.size(); ++i )
        printf("%d ", clauseLits[i]);
    printf("0\n");

    // Add clause to solver
    for ( unsigned i = 0; i < clauseLits.size(); ++i )
        _cadical.add( clauseLits[i] );
    _cadical.add( 0 );

    printf("[DA][SAT]   Clause successfully added.\n");
    printf("[DA][SAT] ================================================\n\n");

    debugPrintSatClauses();
}

void DependencyAnalyzer::_encodeDependencyToClauseLits( const Dependency &dep,
                                                        Vector<int> &outClause )
{
    ASSERT( outClause.empty());

    const auto &vars   = dep.getVars();
    const auto &states = dep.getStates();

    ASSERT( vars.size() == states.size() );
    ASSERT( !vars.empty() );

    /*
      Each Dependency is a nogood: the pattern
        (vars[0]=states[0]) ∧ ... ∧ (vars[k-1]=states[k-1])
      is forbidden.

      SAT encoding: at least one of these literals must be false, so we add:
        (¬ℓ_0 ∨ ¬ℓ_1 ∨ ... ∨ ¬ℓ_{k-1})

      where ℓ_i = phaseToLit( vars[i], states[i] ).
    */
    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        unsigned  reluVar = vars[i];      // Marabou variable ID of this ReLU
        ReLUState phase   = states[i];    // Active / Inactive

        // Literal meaning "reluVar is in phase 'phase'"
        int lit = phaseToLit( reluVar, phase );

        // Nogood ⇒ disjunction of negated literals
        outClause.append( -lit );
    }
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
    addDependenciesTocadical(d);
    Bitmask depMask = _buildDependencyBitmask(vars);
    _recordMinimalDependencyBitmask(depMask);

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

    const unsigned dim = w_t.size();
    ASSERT( dim > 0 );

    // 1) Baseline: box min/max without using the equality w_o·x + b_o = 0
    double baseMin = 0.0;
    double baseMax = 0.0;
    _boxMinMax( w_t, b_t, L, U, baseMin, baseMax );

    double bestMin = baseMin;  // we want largest possible lower bound
    double bestMax = baseMax;  // we want smallest possible upper bound

    // 2) Try all pivots k where w_o[k] != 0 AND w_t[k] != 0
    //
    // If w_t[k] == 0, eliminating x_k via the equality does not change y_t:
    //   y_t = w_t·x + b_t still does not depend on x_k.
    // Our approximate method does not constrain x_k back to [L_k, U_k], so
    // these pivots cannot give anything better than the baseline; we can skip them.
    for ( unsigned k = 0; k < dim; ++k )
    {
        const double denom = w_o[k];
        if ( FloatUtils::isZero( denom ) )
            continue;

        if ( FloatUtils::isZero( w_t[k] ) )
            continue;

        // 2a) Constant term after eliminating x_k:
        //
        // x_k = -(b_o + sum_{j != k} w_o[j] x_j) / w_o[k]
        // y_t = w_t·x + b_t
        //     = sum_{j!=k} [w_t[j] - w_t[k]*(w_o[j]/w_o[k])] x_j  +  b_t - w_t[k]*(b_o/w_o[k])
        //
        double bPrime = b_t - ( w_t[k] * ( b_o / denom ) );

        double mn_k = bPrime;
        double mx_k = bPrime;

        // 2b) Coefficients after substitution and their box contributions,
        //     computed on the fly (no temporary vectors, no _boxMinMax call).
        for ( unsigned j = 0; j < dim; ++j )
        {
            if ( j == k )
                continue;

            // coeff_j = w_t[j] - w_t[k] * (w_o[j] / w_o[k])
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

        // 2c) Update best lower bound: we want the largest mn_k
        if ( FloatUtils::gt( mn_k, bestMin ) )
            bestMin = mn_k;

        // 2d) Update best upper bound: we want the smallest mx_k
        if ( FloatUtils::lt( mx_k, bestMax ) )
            bestMax = mx_k;
    }

    outMin = bestMin;
    outMax = bestMax;
}

bool DependencyAnalyzer::detectAndRecordConflict(
    unsigned layerIndex,
    const std::vector<unsigned> &neurons )
{
    const size_t k = neurons.size();
    ASSERT( k == 2 || k == 3 || k == 4 );

    const NLR::Layer *weightedSumLayer =
        _networkLevelReasoner->getLayer( layerIndex );
    ASSERT( weightedSumLayer );

    // --- Map neurons -> original vars ---
    std::vector<unsigned> vars;
    vars.reserve( k );

    for ( unsigned n : neurons )
    {
        unsigned var = weightedSumLayer->neuronToVariable( n );
        unsigned originalVar = _baseIpqPreprocessor.getOldIndex( var );
        vars.push_back( originalVar );
    }

    // --- Dependency subset pruning ---
    int dependencySubsetMode = 1; // keep your mode logic

    bool issuperset = false;

    if ( dependencySubsetMode == 1 )
    {
        Bitmask depMask = _buildDependencySubBitmask( vars );
        issuperset = _isNonMinimalDependency( depMask );
    }
    else if ( dependencySubsetMode == 2 )
    {
        issuperset = _isSupersetOfKnownDependency( vars );
    }
    else
    {
        ASSERT( false && "Invalid dependencySubsetMode" );
    }

    if ( issuperset )
        return false;

    ASSERT( neurons.size() >= 2 && neurons.size() <= 4 );

    Dependency d;
    bool found = analyzeConflict( layerIndex, neurons, d );

    if ( !found )
        return false;

    return recordConflict( std::move( d ) );
}

#ifdef ENABLE_GUROBI
void DependencyAnalyzer::_buildLpSliceModelMEq(
    GurobiWrapper &lp,
    const Vector<Vector<double>> &w_eq,
    const Vector<double> &b_eq,
    const Vector<double> &L, const Vector<double> &U,
    Vector<String> &varNames ) const
{
    const unsigned dim = L.size();
    const unsigned m   = w_eq.size();
    ASSERT( m == b_eq.size() );

    lp.resetModel();

    varNames = Vector<String>( dim );
    for ( unsigned j = 0; j < dim; ++j )
    {
        String name = Stringf( "z_%u", j );
        varNames[j] = name;
        lp.addVariable( name, L[j], U[j], GurobiWrapper::CONTINUOUS );
    }

    for ( unsigned i = 0; i < m; ++i )
    {
        List<GurobiWrapper::Term> eqTerms;
        for ( unsigned j = 0; j < dim; ++j )
            if ( !FloatUtils::isZero( w_eq[i][j] ) )
                eqTerms.append( GurobiWrapper::Term( w_eq[i][j], varNames[j] ) );

        lp.addEqConstraint( eqTerms, -b_eq[i] );
    }
}
#endif
#ifdef ENABLE_GUROBI
bool DependencyAnalyzer::lpSliceMEqMinMax(
    const Vector<double> &w_t, double b_t,
    const Vector<Vector<double>> &w_eq,
    const Vector<double> &b_eq,
    const Vector<double> &L, const Vector<double> &U,
    double &outMin, double &outMax ) const
{
    ASSERT( w_t.size() == L.size() && L.size() == U.size() );

    GurobiWrapper &lp = _lpReusable;

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
    // --- MIN ---
    lp.setCost( objTerms, b_t );   // minimize
    lp.solve();
    if ( lp.infeasible() || !lp.haveFeasibleSolution() )
    {
        outMin = FloatUtils::infinity();
        isOk = false;
    }
    else
        outMin = lp.getOptimalCostOrObjective();

    // --- MAX ---
    // IMPORTANT: this must only change objective, not rebuild model.
    lp.setObjective( objTerms, b_t ); // maximize
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

void DependencyAnalyzer::_sliceMinMax_givenMEqZero_LP(
    const Vector<double> &w_t, double b_t,
    const Vector<Vector<double>> &w_eq,
    const Vector<double> &b_eq,
    const Vector<double> &L, const Vector<double> &U,
    double &outMin, double &outMax ) const
{
    double mn = 0.0, mx = 0.0;

    bool ok = lpSliceMEqMinMax(
        w_t, b_t,
        w_eq, b_eq,
        L, U,
        mn, mx
    );

    if ( !ok )
    {
        printf( "[DA][LP-SLICE][warning] LP slice infeasible!\n" );
    }

    outMin = mn;
    outMax = mx;
}

#endif




bool DependencyAnalyzer::notifyNeuronFixed( unsigned newVar, ReLUState state )
{
    // TODO: reorganize this function.
    ASSERT( _seenPhase );
    ASSERT( _preprocessor );
    
    // Allow: (1) originally unstable vars, OR
    //        (2) stable var that become exactly at zero.
    //
    // Case (2):
    //    state = Inactive  AND ub == 0
    //    state = Active    AND lb == 0
    //
    printf("[DA][notifyNeuronFixed] start");
    unsigned var = _preprocessor->getOldIndex(newVar); // oldVar
    unsigned newBQVar = _baseIpqPreprocessor.getNewIndex(var);
    printf("[DA][notifyNeuronFixed] start 2");
    double lb = _preprocessedQuery->getLowerBound( newBQVar );
    double ub = _preprocessedQuery->getUpperBound( newBQVar );
    printf("[DA][notifyNeuronFixed] start 3");
    
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
    ReLURuntimeState incoming =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Active
                                       : ReLURuntimeState::Inactive;

    const ReLURuntimeState opposite =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Inactive
                                       : ReLURuntimeState::Active;

    auto ait = _seenPhase->find( var ); 
    if ( ait != _seenPhase->end() && (*ait).second == opposite )
    {
        incoming = ReLURuntimeState::Zero;
        printf("[DA][debug] notifyNeuronFixed(var=%u) Inactive and Active -> Zero.\n", var);
    }
    ASSERT (ait == _seenPhase->end() || (*ait).second == opposite)
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
            const char *rtStr =
                (rt == ReLURuntimeState::Active)   ? "Active"   :
                (rt == ReLURuntimeState::Inactive) ? "Inactive" :
                (rt == ReLURuntimeState::Zero)     ? "Zero"     :
                                                    "Unstable";
            printf("[DA][notifyNeuronFixed]   var=%u  -> %s\n", trackedVar, rtStr);

        }
    }
    else
    {
        printf("[DA][debug] _seenPhase is empty.\n");
    }
    /**************** End Debug ****************/
    return true;
}

void DependencyAnalyzer::notifyLowerBoundUpdate( unsigned newVar,
                                                 double previousLowerBound,
                                                 double newLowerBound )
{
    ASSERT( _preprocessor );
    // Lower bounds must only move up (monotone tightening)
    if ( !FloatUtils::gt( newLowerBound, previousLowerBound ) )
        return;
    ASSERT( !FloatUtils::lt( newLowerBound, previousLowerBound ) );

    unsigned variable = _preprocessor->getOldIndex(newVar);

    // Detect transition to guaranteed Active
    printf("[DA] LB %u (new %u) for %.6g -> %.6g\n", variable, newVar, previousLowerBound, newLowerBound);
    if ( previousLowerBound <= 0.0 && newLowerBound >= 0.0 ){
        printf("[DA] Valid LB\n");
        notifyNeuronFixed( newVar, ReLUState::Active );}
}

void DependencyAnalyzer::notifyUpperBoundUpdate( unsigned newVar,
                                                 double previousUpperBound,
                                                 double newUpperBound )
{
    ASSERT( _preprocessor );
    // Upper bounds must only move down (monotone tightening)
    if ( !FloatUtils::lt( newUpperBound, previousUpperBound ) )
        return;
    ASSERT( !FloatUtils::gt( newUpperBound, previousUpperBound ) );

    unsigned variable = _preprocessor->getOldIndex(newVar);

    // Detect transition to guaranteed Inactive
    printf("[DA] UB %u (new %u) for %.6g -> %.6g\n", variable, newVar, previousUpperBound, newUpperBound);
    if ( previousUpperBound >= 0.0 && newUpperBound <= 0.0 ){
        printf("[DA] Valid UB\n");
        notifyNeuronFixed( newVar, ReLUState::Inactive );}
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
        const unsigned newVar = weightedSumLayer->neuronToVariable( i ); // is this updated to new or old?
        
        const unsigned oldVar = _baseIpqPreprocessor.getOldIndex(newVar);
        ASSERT( newVar == oldVar); // According to Layer::eliminateVariable we never eliminate an input variable
        unsigned var = oldVar;

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
    _preprocessor = nullptr;
    _seenPhase = nullptr;
    _dependencyStates.clear();

    // computeDependencies(); # todo in the setcontex
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
                unsigned newVar = layer->neuronToVariable( neuronIndex );
                unsigned var = _baseIpqPreprocessor.getOldIndex( newVar );
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
    ASSERT( _preprocessor ); 

    if ( _unstableNeurons.empty() )
    {
        printf("[DA][syncWithEnginePreprocessedQuery] no unstable neurons recorded\n");
        return;
    }

    printf("[DA][syncWithEnginePreprocessedQuery] syncing %zu unstable vars with Engine PQ\n",
           _unstableNeurons.size());

    for ( unsigned oldVar : _unstableNeurons )
    {
        unsigned var = _preprocessor->getNewIndex(oldVar);
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

void DependencyAnalyzer::_initializeSatSolver()
{
    printf("[DA][SAT] _initializeSatSolver: this=%p, _cadical=%p\n",
           (void *)this, (void *)&_cadical);

    // Reserve index 0 so that SAT variables 1..N map naturally to indices
    // 1..N in _satVarToReluIndex.
    //
    // We do not clear existing state here; this is called once from the
    // constructor.
    _satVarToReluIndex.append( (unsigned)-1 );
}

ReLURuntimeState DependencyAnalyzer::_getReluPhase( unsigned reluVar ) const
{
    ASSERT( _seenPhase );

    auto it = _seenPhase->find( reluVar );
    if ( it == _seenPhase->end() )
        return ReLURuntimeState::Unstable; // not fixed yet

    // CDHashMap stores ReLURuntimeState as the value type
    ReLURuntimeState state = (*it).second;

    // For the SAT encoding we only care about Active/Inactive;
    // treat Zero like Unstable for now.
    if ( state == ReLURuntimeState::Zero )
        return ReLURuntimeState::Unstable;

    return state;
}

unsigned DependencyAnalyzer::reluIndexToSatVar(unsigned reluVar) const
{
    auto it = _reluIndexToSatVar.find(reluVar);
    if (it == _reluIndexToSatVar.end())
        return 0;  // 0 means "not mapped"
    return it->second;
}
unsigned DependencyAnalyzer::reluIndexToSatVarForce(unsigned reluVar)
{
    unsigned existing = reluIndexToSatVar(reluVar);
    if (existing != 0)
        return existing;

    return _createNewSatVarForRelu(reluVar);
}
unsigned DependencyAnalyzer::_createNewSatVarForRelu(unsigned reluVar)
{
    unsigned newSatVar = _satVarToReluIndex.size();  // SAT vars start at 1
    ASSERT(newSatVar > 0);  // sanity

    printf("[DA][SAT] Assigning SAT var %u to ReLU var %u\n", newSatVar, reluVar);

    _reluIndexToSatVar[reluVar] = newSatVar;
    _satVarToReluIndex.append(reluVar);

    return newSatVar;
}

unsigned DependencyAnalyzer::satVarToReluIndex( unsigned satVar ) const
{
    ASSERT( satVar > 0 );
    ASSERT( satVar < _satVarToReluIndex.size() );

    return _satVarToReluIndex[satVar];
}


int DependencyAnalyzer::phaseToLit( unsigned reluVar,
                                    ReLUState phase )
{
    ASSERT( phase == ReLUState::Active ||
            phase == ReLUState::Inactive );

    unsigned satVar = reluIndexToSatVarForce( reluVar );
    int lit = (phase == ReLUState::Active) ? (int)satVar : -(int)satVar;

    printf("[DA][SAT] phaseToLit: reluVar=%u, phase=%s -> lit=%d\n",
           reluVar,
           (phase == ReLUState::Active ? "Active" : "Inactive"),
           lit);

    return lit;
}

bool DependencyAnalyzer::litToPhase( int lit,
                                     unsigned &reluVar,
                                     ReLUState &phase )
{
    ASSERT( lit != 0 );

    unsigned satVar = (lit > 0) ? (unsigned)lit : (unsigned)(-lit);
    ASSERT( satVar < _satVarToReluIndex.size() );

    reluVar = satVarToReluIndex( satVar );
    phase   = (lit > 0) ? ReLUState::Active : ReLUState::Inactive;

    printf("[DA][SAT] litToPhase: lit=%d -> satVar=%u, reluVar=%u, phase=%s\n",
           lit, satVar, reluVar,
           (phase == ReLUState::Active ? "Active" : "Inactive"));

    return true;
}

void DependencyAnalyzer::_emitTighteningsForImpliedPhase( unsigned reluVar,
                                                          ReLUState impliedPhase,
                                                          List<Tightening> &tightenings )
{
    ASSERT( _preprocessedQuery );

    // Current bounds for this pre-activation variable in the analyzer’s
    // preprocessed query.
    unsigned reluVarBase = _baseIpqPreprocessor.getNewIndex( reluVar );
    double lb = _preprocessedQuery->getLowerBound( reluVarBase );
    double ub = _preprocessedQuery->getUpperBound( reluVarBase );

    printf("[DA][_emitTighteningsForImpliedPhase] old var %u, base-preprocessed var %u\n",
           reluVar, reluVarBase);

    if ( impliedPhase == ReLUState::Active )
    {
        // Active ⇒ pre-activation >= 0
        const double newLb = 0.0;

        // Safety: new LB cannot exceed current UB
        ASSERT( !FloatUtils::gt( newLb, ub ) );

        // If this does not strengthen the LB, skip emitting a tightening
        ASSERT( FloatUtils::gt( newLb, lb ) );

        unsigned engineVar = _preprocessor->getNewIndex( reluVar );

        Tightening t( engineVar, newLb, Tightening::LB );
        tightenings.append( t );

        printf("[DA][SAT]   -> emit LB tightening: x%u >= %.10g (engine var %u)\n",
               reluVar, newLb, engineVar );
    }
    else
    {
        // impliedPhase == ReLUState::Inactive
        ASSERT( impliedPhase == ReLUState::Inactive );
        const double newUb = 0.0;

        // Safety: new UB cannot go below current LB
        ASSERT( !FloatUtils::lt( newUb, lb ) );

        // If this does not strengthen the UB, skip
        ASSERT( FloatUtils::lt( newUb, ub ) );

        unsigned engineVar = _preprocessor->getNewIndex( reluVar );

        Tightening t( engineVar, newUb, Tightening::UB );
        tightenings.append( t );

        printf("[DA][SAT]   -> emit UB tightening: x%u <= %.10g (engine var %u)\n",
               reluVar, newUb, engineVar );
    }
}

void DependencyAnalyzer::getImpliedTighteningsFromSat( List<Tightening> &tightenings )
{
    debugPrintSatClauses();
    printf("\n[DA][SAT] ===== getImpliedTighteningsFromSat =====\n");

    // 3.1 Build assumptions from _seenPhase and call assume inline
    // We iterate over all ReLUs that were ever seen in SAT mapping.
    for ( const auto &entry : _reluIndexToSatVar )
    {
        unsigned reluVar = entry.first;   // Marabou var id for this ReLU
        ReLURuntimeState rt = _getReluPhase( reluVar );

        if ( rt == ReLURuntimeState::Active )
        {
            int lit = phaseToLit( reluVar, ReLUState::Active );
            _cadical.assume( lit );
            printf("[DA][SAT]   assume: var %u is Active (lit %d)\n", reluVar, lit );
        }
        else if ( rt == ReLURuntimeState::Inactive )
        {
            int lit = phaseToLit( reluVar, ReLUState::Inactive );
            _cadical.assume( lit );
            printf("[DA][SAT]   assume: var %u is Inactive (lit %d)\n", reluVar, lit );
        }
        else
        {
            // Zero / Unstable: no assumption for SAT.
        }
    }

    // 3.2 Call propagate and handle the result
    int res = _cadical.propagate();

    if ( res == 20 )
    {
        printf("[DA][SAT]   propagate() returned CONFLICT (20); no tightenings.\n");
        printf("[DA][SAT] ===== end getImpliedTighteningsFromSat (conflict) =====\n\n");
        return;
    }

    if ( res == 10 )
        printf("[DA][SAT]   propagate() returned SAT (10): model under assumptions.\n");
    else if ( res == 0 )
        printf("[DA][SAT]   propagate() returned UNKNOWN/partial (0): trail only.\n");
    else
        printf("[DA][SAT]   propagate() returned %d (unexpected but continuing).\n", res );

    // 3.3 Fetch implicants and convert to implied phases
    std::vector<int> implicants;
    _cadical.get_entrailed_literals( implicants );

    printf("[DA][SAT]   got %zu entailed literals from CaDiCaL\n", implicants.size());

    for ( int lit : implicants )
    {
        unsigned  reluVar = 0;
        ReLUState impliedPhase;

        bool success = litToPhase( lit, reluVar, impliedPhase );
        ASSERT( success );

        ReLURuntimeState currentRt = _getReluPhase( reluVar );

        const char *impliedStr =
            (impliedPhase == ReLUState::Active) ? "Active" : "Inactive";
        const char *rtStr =
            (currentRt == ReLURuntimeState::Active)   ? "Active"   :
            (currentRt == ReLURuntimeState::Inactive) ? "Inactive" :
            (currentRt == ReLURuntimeState::Zero)     ? "Zero"     :
                                                        "Unstable";

        printf("[DA][SAT]   lit %d ⇒ var %u implied %s; current runtime state = %s\n",
               lit, reluVar, impliedStr, rtStr);

        // Skip if already fixed to the same phase
        if ( (currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Active) ||
             (currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Inactive) )
        {
            printf("[DA][SAT]     -> already fixed to that phase, skipping\n");
            continue;
        }

        // If we detect a logical contradiction at runtime (Active vs Inactive), just log it.
        if ( (currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Inactive) ||
             (currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Active) )
        {
            printf("[DA][SAT]     -> WARNING: implied phase contradicts current runtime state, skipping\n");
            ASSERT( false );
        }

        if ( currentRt == ReLURuntimeState::Zero &&
             (impliedPhase == ReLUState::Active || impliedPhase == ReLUState::Inactive) )
        {
            printf("[DA][SAT]     -> WARNING: Zero state with non-zero implied phase, skipping\n");
            ASSERT( false );
        }

        // For Unstable we treat this as a *new* implied phase that can tighten bounds.
        _emitTighteningsForImpliedPhase( reluVar, impliedPhase, tightenings );
    }

    printf("[DA][SAT] ===== end getImpliedTighteningsFromSat (emitted %u tightenings) =====\n\n",
           tightenings.size());
}

void DependencyAnalyzer::debugPrintSatClauses()
{
    return; // disable for now
    printf("[DA][SAT] debugPrintSatClauses: this=%p, _cadical=%p\n",
           (void *)this, (void *)&_cadical);

    const char *path = "debug_cadical.cnf";

    printf("\n[DA][SAT] Dumping CaDiCaL CNF to stdout (via temp file)\n");

    const char *err = _cadical.write_dimacs( path );
    if ( err )
    {
        printf("[DA][SAT]   ERROR from write_dimacs: %s\n", err );
        return;
    }

    FILE *fp = fopen( path, "r" );
    if ( !fp )
    {
        printf("[DA][SAT]   ERROR: cannot open '%s' for reading\n", path );
        return;
    }

    printf("[DA][SAT]   --- Begin DIMACS CNF ---\n");
    char buffer[4096];
    while ( fgets( buffer, sizeof( buffer ), fp ) )
        printf("%s", buffer );
    printf("[DA][SAT]   --- End DIMACS CNF ---\n\n");

    fclose( fp );
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
// runBoundTightening();
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
