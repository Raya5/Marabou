#ifndef __DependencyCalculator_h__
#define __DependencyCalculator_h__

#include <vector>
#include <memory>

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
    struct Stats
    {
        unsigned numLayersVisited = 0;
        unsigned numWeightedSumLayers = 0;

        unsigned totalUnstable = 0;        // sum over layers (as observed)
        unsigned totalCandidates = 0;      // e.g., number of tested pairs (optional)
        unsigned totalDependencies = 0;    // dependencies confirmed

        // Per-layer reporting can be printed directly or collected here later if you want.
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

private:
    IncrementalConflictAnalyser &_ica;

    const Query *_engineQuery;                         // Engine-owned preprocessed query
    NLR::NetworkLevelReasoner *_nlrEngine;        // Provided from Engine/ICA
    NLR::NetworkLevelReasoner *_nlrFromQuery;     // engineQuery->getNetworkLevelReasoner()

    Preprocessor *_preprocessor;
    BoundManager *_boundManager;                       // optional sanity

    std::vector<Dependency> _dependencies;
    Stats _stats;
};

#endif // __DependencyCalculator_h__
