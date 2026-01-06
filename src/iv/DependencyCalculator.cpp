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
{
    // no code yet
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
        const unsigned size = layer->getSize();

        // Placeholder unstable count for now
        std::vector<unsigned> unstable;
        _collectUnstableNeurons( layerIndex, unstable );
        const unsigned numUnstable = unstable.size();

        printf( "[DepCalc] layer=%u type=%u size=%u unstable=%u\n",
                layerIndex, (unsigned)type, size, numUnstable );

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

    printf( "[DC] layer=%u unstable=%zu\n", layerIndex, unstable.size() );

    // this is what you’re missing:
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

    // Sync NLR internal bound store with the current Engine query
    // (same idea as in DependencyAnalyzer)
    _nlrEngine->obtainCurrentBounds( *_engineQuery );

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
    {
        printf( "[DC][fake] layer=%u: unstable<2, skip\n", layerIndex );
        return;
    }

#if DEBUG_FAKE_DEPS
    // Pick the first two unstable neurons (deterministic)
    unsigned n0 = unstableNeurons[0];
    unsigned n1 = unstableNeurons[1];
    if ( n1 < n0 )
        std::swap( n0, n1 );

    const NLR::Layer *layer = _nlrEngine->getLayer( layerIndex );
    ASSERT( layer );
    ASSERT( _preprocessor );

    // neuron index -> newVar (engine indexing) -> oldVar (original indexing)
    const unsigned newVar0 = layer->neuronToVariable( n0 );
    const unsigned newVar1 = layer->neuronToVariable( n1 );

    const unsigned oldVar0 = _preprocessor->getOldIndex( newVar0 );
    const unsigned oldVar1 = _preprocessor->getOldIndex( newVar1 );

    std::vector<unsigned> vars = { oldVar0, oldVar1 };
    if ( vars[1] < vars[0] )
        std::swap( vars[0], vars[1] );

    printf( "[DC][fake] layer=%u try pair oldVars=(%u,%u)\n",
            layerIndex, vars[0], vars[1] );

    // EARLY PRUNE via ICA (vars-only)
    if ( _ica.isNonMinimalDependencyVarsSubMask( vars ) )
    {
        printf( "[DC][fake] layer=%u pruned pair oldVars=(%u,%u)\n",
                layerIndex, vars[0], vars[1] );
        return;
    }

    // Fake forbidden assignment pattern: (A, I)
    std::vector<bool> isActive = { true, false };

    // Record immediately into ICA: minimality + conflict storage + encode into CaDiCaL
    _ica.addDependency( vars, isActive );

    printf( "[DC][fake] layer=%u recorded dep->conflict oldVars=(%u,%u) pattern=(A,I)\n",
            layerIndex, vars[0], vars[1] );

    // Optional stats
    _stats.totalCandidates += 1;
    _stats.totalDependencies += 1;

    // Only one fake dependency per layer for now
    return;
#else
    (void)layerIndex;
    (void)unstableNeurons;
    return;
#endif
}


bool DependencyCalculator::_pruneByKnownMinimalVarSets( const std::vector<unsigned> &oldVars ) const
{
    return _ica.isNonMinimalDependencyVarsSubMask( oldVars );
}



bool DependencyCalculator::_analyzeNeuronSet( unsigned /*layerIndex*/,
                                             const std::vector<unsigned> &neuronsSorted,
                                             Dependency &outDependency ) const
{
    (void) neuronsSorted;
    (void) outDependency;
    // no code yet
    return false;
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
