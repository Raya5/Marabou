#ifndef __DependencyCalculator_h__
#define __DependencyCalculator_h__

#include <vector>
#include <memory>
#include "Vector.h"
#include "GurobiWrapper.h"

class Query;
class Preprocessor;
class BoundManager;

namespace NLR {
class NetworkLevelReasoner;
class Layer;
}

class IncrementalConflictAnalyser;
class Dependency;

class DependencyCalculator
{
public:

    struct LayerDepStats
    {
        unsigned layerIndex = 0;        // NLR layer index
        unsigned wsOrdinal = 0;         // 0 for first WS layer, 1 for second, ...
        unsigned unstableCount = 0;
        unsigned depsFound = 0;
        double secondsSpent = 0.0;
    };

    struct Stats
    {
        unsigned numLayersVisited = 0;
        unsigned numWeightedSumLayers = 0;

        unsigned totalUnstable = 0;
        unsigned totalCandidates = 0;
        unsigned totalDependencies = 0;
        unsigned totalPruned = 0;

        std::vector<LayerDepStats> perWsLayer;  // one entry per WEIGHTED_SUM layer
    };

    /*
      Construct a calculator for a single solve call.

      Ownership:
        - ICA is borrowed (must outlive this object; it will, because this object is stack-local).
        - Query/NLR/Preprocessor/BoundManager are borrowed (Engine-owned).
    */
    DependencyCalculator( IncrementalConflictAnalyser &ica,
                          const Query *engineQuery,
                          NLR::NetworkLevelReasoner *nlrFromEngine,
                          Preprocessor *preprocessor,
                          BoundManager *boundManager );

    /*
      Run dependency discovery.
      Returns a list of Dependencies (old-variable indexing).
      Caller (ICA) will translate each Dependency into a conflict and store it with epsilon.
    */
    void run();

    const Stats &getStats() const;

private:
    // ---- Core driver ----
    void _scanAllLayers();
    void _scanWeightedSumLayer( unsigned layerIndex );

    // ---- Unstable collection ----
    void _collectUnstableNeurons( unsigned layerIndex,
                                 std::vector<unsigned> &unstableNeurons ) const;

    // ---- Candidate enumeration ----
    void _enumeratePairs( unsigned layerIndex,
                          const std::vector<unsigned> &unstableNeurons );

    // (Later) triples/quads hooks:
    // void _enumerateTriples(...);
    // void _enumerateQuads(...);

    // ---- Minimality pruning (vars-only) ----
    // Build mask over SAT-vars (no polarity). Used for early pruning.
    // Two variants: full vs sub (sub skips unmapped SAT vars).
    // These call ICA helpers (or use ICA’s SAT mapping).
    bool _isPrunedByIcaMinimalVarSets( const std::vector<unsigned> &oldVars ) const;
    bool _pruneByKnownMinimalVarSets( const std::vector<unsigned> &oldVars ) const;

    // ---- Conflict analysis (expensive part) ----
    // Try to derive a forbidden phase assignment for the given neuron set.
    // If successful, append a Dependency (old vars + forbidden states).
    bool _analyzeNeuronSet( unsigned layerIndex,
                            const std::vector<unsigned> &neuronsSorted,
                            Dependency &outDependency ) const;

    // ---- Sanity checks ----
    void _assertNlrConsistency() const;
    void _assertBoundsConsistencyForVar( unsigned newVar ) const; // optional: query vs BM

    // ---- Bounds and slicing helpers ---- 
    void _getLayerBounds( const NLR::Layer *layer, Vector<double> &lowerBounds, Vector<double> &upperBounds ) const;
    void _boxMinMax( const Vector<double> &a, double b, const Vector<double> &L, const Vector<double> &U, double &outMin, double &outMax ) const;
    void _sliceMinMax_givenOtherZero( const Vector<double> &w_t, double b_t, const Vector<double> &w_o, double b_o, const Vector<double> &L, const Vector<double> &U, double &outMin, double &outMax ) const;

    // ---- Gurobi / LP slicing (only if you want k>=3 working) ---- 
    void _sliceMinMax_givenMEqZero_LP( const Vector<double> &w_t, double b_t, const Vector<Vector<double>> &w_eq, const Vector<double> &b_eq, const Vector<double> &L, const Vector<double> &U, double &outMin, double &outMax ) const;
    bool _lpSliceMEqMinMax( const Vector<double> &w_t, double b_t, const Vector<Vector<double>> &w_eq, const Vector<double> &b_eq, const Vector<double> &L, const Vector<double> &U, double &outMin, double &outMax ) const;
    void _buildLpSliceModelMEq( GurobiWrapper &lp, const Vector<Vector<double>> &w_eq, const Vector<double> &b_eq, const Vector<double> &L, const Vector<double> &U, Vector<String> &varNames ) const;
        
private:
    IncrementalConflictAnalyser &_ica;

    const Query *_engineQuery;                         // Engine-owned preprocessed query
    NLR::NetworkLevelReasoner *_nlrEngine;        // Provided from Engine/ICA
    NLR::NetworkLevelReasoner *_nlrFromQuery;     // engineQuery->getNetworkLevelReasoner()

    Preprocessor *_preprocessor;
    BoundManager *_boundManager;                       // optional sanity

    std::vector<Dependency> _dependencies;

    mutable GurobiWrapper _lpReusable;
    mutable bool _lpReusableInitialized;

    Stats _stats;
};

#endif // __DependencyCalculator_h__
