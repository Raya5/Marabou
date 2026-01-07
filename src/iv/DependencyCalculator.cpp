#include "DependencyCalculator.h"

// includes you’ll need later:
#include "IncrementalConflictAnalyser.h"
#include "Query.h"
#include "Preprocessor.h"
#include "BoundManager.h"
#include "NetworkLevelReasoner.h"
#include "Dependency.h"
#include "FloatUtils.h"
// <algorithm> <cstdio> etc.
// #include "FloatUtils.h"
#include <cstdio>
#include <algorithm>
#include <chrono>

#include "Query.h"
#include "FloatUtils.h"
#include "NetworkLevelReasoner.h"   // or the right header for NLR
#include <algorithm>

DependencyCalculator::DependencyCalculator( IncrementalConflictAnalyser &ica,
                                           const Query *engineQuery,
                                           NLR::NetworkLevelReasoner *nlrFromEngine,
                                           Preprocessor *preprocessor,
                                           BoundManager *boundManager )
    : _ica( ica )
    , _engineQuery( engineQuery )
    , _nlrEngine( nlrFromEngine )
    , _nlrFromQuery( nullptr )
    , _preprocessor( preprocessor )
    , _boundManager( boundManager )
    , _lpReusableInitialized( false )
{
}

void DependencyCalculator::run()
{
    _assertNlrConsistency();
    _scanAllLayers();
}


const DependencyCalculator::Stats &DependencyCalculator::getStats() const
{
    return _stats;
}

// --- private skeletons ---

void DependencyCalculator::_scanAllLayers()
{
    ASSERT( _engineQuery );
    ASSERT( _nlrEngine );

    const NLR::NetworkLevelReasoner *nlrFromQuery =
        _engineQuery->getNetworkLevelReasoner();

    ASSERT( nlrFromQuery );
    ASSERT( nlrFromQuery == _nlrEngine );

    _nlrEngine->obtainCurrentBounds( *_engineQuery );

    const unsigned numLayers = _nlrEngine->getNumberOfLayers();
    _stats.numLayersVisited = numLayers;

    printf( "[DepCalc] scanning %u layers\n", numLayers );

    for ( unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex )
    {
        const NLR::Layer *layer = _nlrEngine->getLayer( layerIndex );
        if ( !layer )
        {
            ASSERT( false );
            printf( "[DepCalc] layer=%u <null>\n", layerIndex );
            continue;
        }

        const auto type = layer->getLayerType();

        if ( type == NLR::Layer::WEIGHTED_SUM )
        {
            _stats.numWeightedSumLayers++;
            _scanWeightedSumLayer( layerIndex );
        }

    }

    printf( "[DepCalc] done: weightedSumLayers=%u\n",
            _stats.numWeightedSumLayers );
}



void DependencyCalculator::_scanWeightedSumLayer( unsigned layerIndex )
{
    ASSERT( _nlrEngine );

    const NLR::Layer *layer = _nlrEngine->getLayer( layerIndex );
    if ( !layer )
        return;

    if ( layer->getLayerType() != NLR::Layer::WEIGHTED_SUM )
        return;

    std::vector<unsigned> unstable;
    _collectUnstableNeurons( layerIndex, unstable );
    
    _stats.totalUnstable += unstable.size();
    printf( "[DC] layer=%u unstable=%zu\n", layerIndex, unstable.size() );

    _enumeratePairs( layerIndex, unstable );
}


void DependencyCalculator::_collectUnstableNeurons( unsigned layerIndex,
                                                    std::vector<unsigned> &unstableNeurons ) const
{
    unstableNeurons.clear();

    ASSERT( _nlrEngine );
    ASSERT( _engineQuery );

    const NLR::Layer *layer = _nlrEngine->getLayer( layerIndex );
    if ( !layer )
        return;

    // We only collect from weighted-sum layers in this prototype
    if ( layer->getLayerType() != NLR::Layer::WEIGHTED_SUM )
        return;

    const unsigned layerSize = layer->getSize();
    unstableNeurons.reserve( layerSize );

    for ( unsigned n = 0; n < layerSize; ++n )
    {
        const unsigned newVar = layer->neuronToVariable( n );

        const double lb = _engineQuery->getLowerBound( newVar );
        const double ub = _engineQuery->getUpperBound( newVar );

        // crosses 0  <=> lb < 0 and ub > 0
        if ( FloatUtils::lt( lb, 0.0 ) && FloatUtils::gt( ub, 0.0 ) )
            unstableNeurons.push_back( n );

        ASSERT( _boundManager );
        {
            const double bmLb = _boundManager->getLowerBound( newVar );
            const double bmUb = _boundManager->getUpperBound( newVar );
            ASSERT( FloatUtils::areEqual( bmLb, lb ) );
            ASSERT( FloatUtils::areEqual( bmUb, ub ) );
        }

    }

    // Make deterministic
    std::sort( unstableNeurons.begin(), unstableNeurons.end() );
}



#define DEBUG_FAKE_DEPS 1

void DependencyCalculator::_enumeratePairs( unsigned layerIndex,
                                           const std::vector<unsigned> &unstableNeurons )
{
    if ( unstableNeurons.size() < 2 )
        return;

    const NLR::Layer *layer = _nlrEngine->getLayer( layerIndex );
    ASSERT( layer );
    ASSERT( _preprocessor );

    unsigned depPairCount = 0;
    for ( size_t i = 0; i + 1 < unstableNeurons.size(); ++i )
    {
        for ( size_t j = i + 1; j < unstableNeurons.size(); ++j )
        {
            const unsigned n0 = unstableNeurons[i];
            const unsigned n1 = unstableNeurons[j];

            // neuron index -> newVar (engine indexing) -> oldVar (original indexing)
            const unsigned newVar0 = layer->neuronToVariable( n0 );
            const unsigned newVar1 = layer->neuronToVariable( n1 );

            const unsigned oldVar0 = _preprocessor->getOldIndex( newVar0 );
            const unsigned oldVar1 = _preprocessor->getOldIndex( newVar1 );

            // Canonical sort (and assert it)
            unsigned a = oldVar0;
            unsigned b = oldVar1;
            ASSERT( a < b );
            if ( b < a )
                std::swap( a, b );

            std::vector<unsigned> oldVars = { a, b };

            // Early prune via ICA minimality (vars-only)
            if ( _ica.isNonMinimalDependencyVarsSubMask( oldVars ) )
            {
                _stats.totalPruned++;
                printf( "[DC] layer=%u pruned pair oldVars=(%u,%u)\n",
                        layerIndex, oldVars[0], oldVars[1] );
                continue;
            }

            _stats.totalCandidates++;


            std::vector<unsigned> neurons = { n0, n1 };
            ASSERT( neurons[0] < neurons[1] );

            Dependency dep;
            if ( _analyzeNeuronSet( layerIndex, neurons, dep ) )
            {
                printf( "[DC] layer=%u found dep oldVars=(%u,%u) forbid=(%u,%u)\n",
                        layerIndex,
                        dep.getVars()[0],
                        dep.getVars()[1],
                        static_cast<unsigned>( dep.getStates()[0] ),
                        static_cast<unsigned>( dep.getStates()[1] ) );
                _stats.totalDependencies++;
                depPairCount++;

                // Later in Phase 4: translate dep -> (oldVars,isActiveList) and call:
                // _ica.addDependency(oldVars, isActiveList);

                const std::vector<unsigned> &oldVars = dep.getVars();
                const std::vector<ReLUState> &states = dep.getStates();
                ASSERT( oldVars.size() == states.size() );

                // Canonical ordering should already hold, but keep the check
                for ( size_t t = 1; t < oldVars.size(); ++t )
                    ASSERT( oldVars[t - 1] < oldVars[t] );

                std::vector<bool> isActiveList;
                isActiveList.reserve( states.size() );
                for ( ReLUState s : states )
                    isActiveList.push_back( s == ReLUState::Active );

                _ica.addDependency( oldVars, isActiveList );

            }

        }
    }
    printf( "[DC] layer=%u , unstable size %zu total dep pairs found=%u\n",
            layerIndex, unstableNeurons.size(), depPairCount );
}


bool DependencyCalculator::_pruneByKnownMinimalVarSets( const std::vector<unsigned> &oldVars ) const
{
    return _ica.isNonMinimalDependencyVarsSubMask( oldVars );
}


// _analyzeNeuronSet(layerIndex, neuronsSorted, outDep):
// 1) Read weighted-sum layer and its single predecessor layer.
// 2) Get predecessor box bounds L,U.
// 3) Build weights/biases (W[i],B[i]) for each neuron in neuronsSorted.
// 4) For each neuron i:
//      compute sliced bounds [Lcond[i],Ucond[i]] of (W[i]·x + B[i])
//      under constraints (W[j]·x + B[j] == 0) for all j != i
//      - k==2: analytic elimination (_sliceMinMax_givenOtherZero)
//      - k>=3: LP slice (_sliceMinMax_givenMEqZero_LP) if ENABLE_GUROBI
// 5) If any slice infeasible → no dependency.
// 6) Derive forced phase for each neuron from sliced bounds:
//      forced Active if Lcond>0, forced Inactive if Ucond<0, require exactly one.
// 7) Forbidden assignment is the opposite of forced.
// 8) Map neurons to oldVars, output Dependency(oldVars, forbiddenStates).
bool DependencyCalculator::_analyzeNeuronSet( unsigned layerIndex,
                                             const std::vector<unsigned> &neuronsSorted,
                                             Dependency &outDependency ) const
{
    const NLR::Layer *weightedSumLayer = _nlrEngine->getLayer( layerIndex );
    ASSERT( weightedSumLayer );
    ASSERT( weightedSumLayer->getLayerType() == NLR::Layer::WEIGHTED_SUM );

    const unsigned k = neuronsSorted.size();
    ASSERT( k == 2 || k == 3 || k == 4 );

    // Validate ordering and bounds
    const unsigned layerSize = weightedSumLayer->getSize();
    for ( unsigned i = 0; i < k; ++i )
    {
        ASSERT( neuronsSorted[i] < layerSize );
        if ( i > 0 )
            ASSERT( neuronsSorted[i - 1] < neuronsSorted[i] );
    }

    // Single predecessor (typical affine layer)
    const auto &sources = weightedSumLayer->getSourceLayers();
    ASSERT( sources.size() == 1 );
    const unsigned prevLayerIndex = sources.begin()->first;

    const NLR::Layer *prevLayer = _nlrEngine->getLayer( prevLayerIndex );
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
        const unsigned n = neuronsSorted[idx];

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
            const unsigned o = ( i == 0 ? 1 : 0 );
            _sliceMinMax_givenOtherZero( W[i], B[i], W[o], B[o],
                                         lowerPrev, upperPrev,
                                         Lcond[i], Ucond[i] );
        }
        else
        {

            Vector<Vector<double>> w_eq( k - 1 );
            Vector<double> b_eq( k - 1 );

            unsigned p = 0;
            for ( unsigned j = 0; j < k; ++j )
            {
                if ( j == i )
                    continue;
                w_eq[p] = W[j];
                b_eq[p] = B[j];
                ++p;
            }

            _sliceMinMax_givenMEqZero_LP( W[i], B[i],
                                          w_eq, b_eq,
                                          lowerPrev, upperPrev,
                                          Lcond[i], Ucond[i] );

        }
    }

    // If any slice is infeasible, ignore
    for ( unsigned i = 0; i < k; ++i )
    {
        if ( !FloatUtils::isFinite( Lcond[i] ) || !FloatUtils::isFinite( Ucond[i] ) )
            return false;
    }

    // Determine forced phases from the sliced bounds
    std::vector<bool> forcedActive( k, false ), forcedInactive( k, false );
    for ( unsigned i = 0; i < k; ++i )
    {
        forcedActive[i]   = FloatUtils::gt( Lcond[i], 0.0 );
        forcedInactive[i] = FloatUtils::lt( Ucond[i], 0.0 );
    }

    auto isForced = []( bool fa, bool fi ) {
        return ( fa && !fi ) || ( fi && !fa );
    };

    // Require every neuron in the set to be forced to a unique phase
    for ( unsigned i = 0; i < k; ++i )
        if ( !isForced( forcedActive[i], forcedInactive[i] ) )
            return false;

    // Build forbidden assignment ("nogood"): opposite of forced
    std::vector<ReLUState> forbid( k, ReLUState::Inactive );
    for ( unsigned i = 0; i < k; ++i )
        forbid[i] = forcedInactive[i] ? ReLUState::Active : ReLUState::Inactive;

    // Map neuron indices to old Marabou variables (old indexing)
    std::vector<unsigned> oldVars;
    oldVars.reserve( k );
    for ( unsigned i = 0; i < k; ++i )
    {
        const unsigned newVar = weightedSumLayer->neuronToVariable( neuronsSorted[i] );
        oldVars.push_back( _preprocessor->getOldIndex( newVar ) );
    }

    // Dependency expects vars sorted ascending aligned with states; neuronsSorted is sorted
    outDependency = Dependency( oldVars, forbid );
    return true;
}


void DependencyCalculator::_assertNlrConsistency() const
{
    ASSERT( _engineQuery );
    ASSERT( _nlrEngine );

    const NLR::NetworkLevelReasoner *fromQuery = _engineQuery->getNetworkLevelReasoner();
    ASSERT( fromQuery );

    // You asked to keep both for now and assert they match
    ASSERT( fromQuery == _nlrEngine );

    // Optional: if you want to ensure successor metadata exists: // Assume it does for now
    // const_cast<NLR::NetworkLevelReasoner *>( fromQuery )->computeSuccessorLayers();
}


void DependencyCalculator::_assertBoundsConsistencyForVar( unsigned /*newVar*/ ) const
{
    // no code yet
}


void DependencyCalculator::_getLayerBounds( const NLR::Layer *layer,
                                           Vector<double> &lowerBounds,
                                           Vector<double> &upperBounds ) const
{
    ASSERT( layer );

    const unsigned n = layer->getSize();

    Vector<double> L;
    Vector<double> U;

    for ( unsigned i = 0; i < n; ++i )
    {
        L.append( layer->getLb( i ) );
        U.append( layer->getUb( i ) );
    }

    lowerBounds = L;
    upperBounds = U;
}


void DependencyCalculator::_boxMinMax( const Vector<double> &a, double b,
                                      const Vector<double> &L,
                                      const Vector<double> &U,
                                      double &outMin,
                                      double &outMax ) const
{
    ASSERT( a.size() == L.size() );
    ASSERT( a.size() == U.size() );

    double mn = b;
    double mx = b;

    const unsigned dim = a.size();
    for ( unsigned j = 0; j < dim; ++j )
    {
        const double aj = a[j];
        if ( aj >= 0.0 )
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
void DependencyCalculator::_sliceMinMax_givenOtherZero(
    const Vector<double> &w_t, double b_t,
    const Vector<double> &w_o, double b_o,
    const Vector<double> &L,
    const Vector<double> &U,
    double &outMin,
    double &outMax ) const
{
    ASSERT( w_t.size() == w_o.size() );
    ASSERT( w_t.size() == L.size() );
    ASSERT( L.size() == U.size() );

    const unsigned dim = w_t.size();
    ASSERT( dim > 0 );

    // Baseline: ignore equality constraint
    double baseMin = 0.0;
    double baseMax = 0.0;
    _boxMinMax( w_t, b_t, L, U, baseMin, baseMax );

    double bestMin = baseMin;
    double bestMax = baseMax;

    // Try eliminating each pivot k
    for ( unsigned k = 0; k < dim; ++k )
    {
        const double denom = w_o[k];
        if ( FloatUtils::isZero( denom ) )
            continue;

        if ( FloatUtils::isZero( w_t[k] ) )
            continue;

        // Eliminate x_k using w_o·x + b_o = 0
        const double bPrime = b_t - ( w_t[k] * ( b_o / denom ) );

        double mn_k = bPrime;
        double mx_k = bPrime;

        for ( unsigned j = 0; j < dim; ++j )
        {
            if ( j == k )
                continue;

            const double coeff =
                w_t[j] - ( w_t[k] * ( w_o[j] / denom ) );

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

        if ( FloatUtils::gt( mn_k, bestMin ) )
            bestMin = mn_k;

        if ( FloatUtils::lt( mx_k, bestMax ) )
            bestMax = mx_k;
    }

    outMin = bestMin;
    outMax = bestMax;
}

void DependencyCalculator::_sliceMinMax_givenMEqZero_LP(
    const Vector<double> &w_t,
    double b_t,
    const Vector<Vector<double>> &w_eq,
    const Vector<double> &b_eq,
    const Vector<double> &L,
    const Vector<double> &U,
    double &outMin,
    double &outMax ) const
{
    // Wrapper helper that fills outMin/outMax; on infeasible slice, returns +/- infinity
    double mn = 0.0;
    double mx = 0.0;

    (void)_lpSliceMEqMinMax( w_t, b_t, w_eq, b_eq, L, U, mn, mx );

    outMin = mn;
    outMax = mx;
}

bool DependencyCalculator::_lpSliceMEqMinMax( const Vector<double> &w_t,
                                            double b_t,
                                            const Vector<Vector<double>> &w_eq,
                                            const Vector<double> &b_eq,
                                            const Vector<double> &L,
                                            const Vector<double> &U,
                                            double &outMin,
                                            double &outMax ) const
{
    // Solve two LPs: minimize and maximize w_t·z + b_t under equality slice constraints
    ASSERT( w_t.size() == L.size() );
    ASSERT( L.size() == U.size() );

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


void DependencyCalculator::_buildLpSliceModelMEq( GurobiWrapper &lp,
                                                 const Vector<Vector<double>> &w_eq,
                                                 const Vector<double> &b_eq,
                                                 const Vector<double> &L,
                                                 const Vector<double> &U,
                                                 Vector<String> &varNames ) const
{
    // Build LP:
    //   variables z_j in [L_j, U_j]
    //   constraints: w_eq[i]·z + b_eq[i] = 0
    //
    // Wrapper expects: sum terms == -b_eq[i]

    const unsigned dim = L.size();
    const unsigned m = w_eq.size();
    ASSERT( m == b_eq.size() );
    ASSERT( dim == U.size() );

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
        {
            if ( !FloatUtils::isZero( w_eq[i][j] ) )
                eqTerms.append( GurobiWrapper::Term( w_eq[i][j], varNames[j] ) );
        }

        lp.addEqConstraint( eqTerms, -b_eq[i] );
    }
}
