/*********************                                                        */
/*! \file DependencyAnalyzer.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Raya E. 
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2025 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** DependencyAnalyzer
 ** -------------------
 ** Built once per incremental batch from a base InputQuery that already has
 ** covering input bounds. It retains a non-owning pointer to that base
 ** InputQuery, and (internally, in the .cpp) constructs a preprocessed
 ** Query (which owns the NetworkLevelReasoner). The analyzer is then shared
 ** across multiple per-point solves to enable future reuse.
 **/
#ifndef __DependencyAnalyzer_h__
#define __DependencyAnalyzer_h__

#include "InputQuery.h"
#include "Query.h"
#include "NetworkLevelReasoner.h"
#include "Layer.h"
#include "Dependency.h"
#include "DependencyState.h"
#include "context/cdo.h"
#include "context/context.h"

// #include <memory>

/**************** For Debugging ********************/
struct BoundsSnapshot {
    struct Entry { double lb, ub; };
    std::unordered_map<unsigned, Entry> byVar; // var -> {lb, ub}
};
/**************** End For Debugging ********************/

class DependencyAnalyzer
{
public:
    /*
      Construct the analyzer from a base InputQuery.
      NOTE: The analyzer does not take ownership; the caller must ensure
            the pointed-to InputQuery outlives this analyzer, or that the
            analyzer doesn’t dereference it after destruction of the base.
    */
    DependencyAnalyzer( const InputQuery *baseIpq,
                        const Vector<Vector<double>> &allLbs,
                        const Vector<Vector<double>> &allUbs );

    ~DependencyAnalyzer();
    /*
      Build internal state (preprocessed query + NLR) from the base IPQ.
      Call once right after construction.
    */
    void buildFromBase();

    /*
      Set the current Context object used for context-dependent
      data structures and backtracking. Should be called once the
      engine context is available.
    */
    void setContext( CVC4::context::Context *ctx );

    /*
      Accessor for the stored base InputQuery pointer (may be nullptr).
      Intended for internal/diagnostic use only.
    */
    const InputQuery *getBaseInputQuery() const;

    /*
      Lightweight diagnostic hook (safe to call anytime)
    */
    void printSummary() const;

    /*
      Runs DeepPoly once and applies the resulting tightenings to the preprocessed query.
      Returns how many bounds got tightened.
    */
    unsigned runBoundTightening();

    /*
      Gather unstable neurons of the given weighted-sum layer from NLR pre-activation bounds
    */
    void collectUnstableNeurons( unsigned layerIndex, std::vector<unsigned> &unstableNeurons ) const;

    /*
      Compute size-2 dependencies (as conflicts) for a given WEIGHTED_SUM layer index.
      Returns number of newly recorded conflicts.
    */
    unsigned computeSameLayerDependencies( unsigned weightedSumLayerIndex );
    unsigned computeSameLayerDependencies(); //TODO: add description, this is for all the layers.

    /*
      Test a single pair (q,r) and record a conflict if one is found. Returns true iff a new conflict was added.
    */
    bool detectAndRecordPairConflict(unsigned layerIndex,
                                 unsigned neuronA, unsigned neuronB);

    /*
      Analyze whether neurons q and r in the same layer form a dependency
      (forbidden ReLU state combination). If found, populate outDependency.

      Returns true if a dependency was detected, false otherwise.
    */
    bool analyzePairConflict( unsigned layerIndex,
                              unsigned q, unsigned r,
                              Dependency &outDependency );

    bool _isSupersetOfKnownDependency(const std::vector<unsigned> &variables) const;

    /*
      Record a discovered dependency in the internal storage for its layer.
      Handles canonicalization and avoids duplicates.

      Returns true if the dependency was newly inserted.
    */
    bool recordConflict( Dependency d );

    /*
      Extract per-neuron lower and upper bounds from the given layer.
    */
    void _getLayerBounds( const NLR::Layer *layer,
                          Vector<double> &lowerBounds,
                          Vector<double> &upperBounds ) const;
                          
    /*
      Return index of largest-magnitude nonzero entry in w (ASSERT if all zero).
    */
    unsigned _argmaxAbsNonzero( const Vector<double> &w ) const;

    /*
      Compute min/max of target (w_t·x + b_t) subject to box L<=x<=U and equality (w_o·x + b_o = 0).
      Uses a single-variable elimination with argmax-abs pivot from w_o.
    */
    void _sliceMinMax_givenOtherZero( const Vector<double> &w_target, double b_target,
                                      const Vector<double> &w_other,  double b_other,
                                      const Vector<double> &L, const Vector<double> &U,
                                      double &outMin, double &outMax ) const;

    /*
      Interval extrema over a box for affine form a·x + b (no equalities).
    */
    void _boxMinMax( const Vector<double> &a, double b,
                    const Vector<double> &L, const Vector<double> &U,
                    double &outMin, double &outMax ) const;

    /*
      Notify the DependencyAnalyzer that a ReLU pre-activation variable
      (identified by its Marabou variable ID) has been fixed to a phase.

      This updates the runtime state for all dependencies that include
      the literal (var = state). It does not yet trigger propagation.
    */
    bool notifyNeuronFixed( unsigned var, ReLUState state );

    /*
      Notify the DependencyAnalyzer that the lower bound of a pre-activation variable
      has been updated. If this tightening crosses zero (i.e., newLowerBound > 0),
      the neuron is fixed to the Active phase.
    */
    void notifyLowerBoundUpdate( unsigned variable,
                                double previousLowerBound,
                                double newLowerBound );

    /*
      Notify the DependencyAnalyzer that the upper bound of a pre-activation variable
      has been updated. If this tightening crosses zero (i.e., newUpperBound < 0),
      the neuron is fixed to the Inactive phase.
    */
    void notifyUpperBoundUpdate( unsigned variable,
                                double previousUpperBound,
                                double newUpperBound );
    /*
        Advance the analyzer to the next query in the schedule.

        This method:
          1. Increments _nextQueryToSolve.
          2. Recomputes the covering input box over all remaining queries.
          3. Generates tightenings for input variables whose bounds shrink.
          4. Applies those tightenings to the preprocessed query.
          5. Resets context-dependent runtime state (phases, dep states).

        Called by Engine after each SAT/UNSAT/UNKNOWN result so that the
        analyzer stays synchronized with the incremental solve cycle.
    */
    void notifyQuerySolved();


    /*
        Add description. 
    */
    void getImpliedTightenings( List<Tightening> &tightenings );

    /*
      Synchronize unstable neurons with the Engine's preprocessed query.

      For each var in _unstableNeurons, read its bounds from engineQuery:
        - if lb >= 0 ⇒ notifyLowerBoundUpdate(var, -inf, lb) ⇒ Active
        - else if ub <= 0 ⇒ notifyUpperBoundUpdate(var, +inf, ub) ⇒ Inactive

      This allows the analyzer to learn about neurons that are already stable
      in the Engine's view at the beginning of the solve.
    */
    void syncWithEnginePreprocessedQuery( const Query &engineQuery );


    /**************** For Debugging ********************/

    BoundsSnapshot snapshotBounds(const std::vector<unsigned> &vars = {});
    std::vector<std::tuple<unsigned,double,double,double,double>>
    diffBounds(const BoundsSnapshot &a, const BoundsSnapshot &b, double eps = 1e-9);
    void printBoundsDiff(const std::vector<std::tuple<unsigned,double,double,double,double>> &diff,
                        unsigned maxItems = 50);
    void debugdiff();

    /**************** End For Debugging ********************/

private:
    CVC4::context::Context *_context;

    unsigned _numQueries;
    unsigned _inputDim; // TODO: maybe we do not need it. check that
    unsigned _nextQueryToSolve;

    /*
      Non-owning pointer to the base InputQuery provided by the builder
    */
    const InputQuery *_baseIpq; // non-owning, read-only pointer (MVP) 
    
    /*
        Per-query input-domain bounds exactly as provided by Python.

        _originalLbs[q][i] = lower bound of input dimension i for query q
        _originalUbs[q][i] = upper bound of input dimension i for query q

        These remain immutable and serve as the source from which covering
        boxes are recomputed as queries are solved.
    */
    const Vector<Vector<double>> _originalLbs;
    const Vector<Vector<double>> _originalUbs;

    /*
      TODO: add decription. 
    */
    Vector<double> _currentLb;
    Vector<double> _currentUb;

    /*
      Preprocessed Query (owns the NLR); created in the .cpp
    */
    std::unique_ptr<Query> _preprocessedQuery;

    /*
      Cached raw pointer to the NLR owned by _preprocessedQuery.
    */
    NLR::NetworkLevelReasoner *_networkLevelReasoner;

    // ---- Dependency storage & runtime ----

    std::vector<Dependency> _dependencies;         // flat store; id = index
    std::vector<DependencyState> _dependencyStates; // runtime, parallel to _dependencies

    // ---- Watches: var -> deps containing that var with the given state ----
    std::unordered_map<unsigned, Vector<DependencyState::DependencyId>> _watchActive;    // Dependencies containing (var, Active)
    std::unordered_map<unsigned, Vector<DependencyState::DependencyId>> _watchInactive;  // Dependencies containing (var, Inactive)

    Vector<DependencyState::DependencyId> _activeDepIds;

    // List of pre-activation variables that are unstable (lb < 0 < ub)
    // at the initial covering box / DeepPoly run.
    std::vector<unsigned> _unstableNeurons;    
    
    /**************** For Debugging ********************/
    /*
      Debug-only dependency index for duplicate assertion.
      Maps each Dependency (value-based) to its index in _dependencies.
      Used to ensure no duplicate insertions during recordConflict().
    */
    std::unordered_map<Dependency, DependencyState::DependencyId, Dependency::Hasher> _dependencyIndex;
    
    // Track last known phase per variable to avoid duplicate notifications
    CVC4::context::CDHashMap<unsigned, ReLURuntimeState, std::hash<unsigned>> *_seenPhase;

    /**************** End For Debugging ********************/


    /*
      Store the given dependency value in _dependencies and update the
      index used for duplicate detection. Returns the assigned id.
    */
    DependencyState::DependencyId _addDependency( const Dependency &d );

    /*
      Create and append a DependencyState runtime object for the given
      dependency id. Uses the current Context to allocate context-dependent
      runtime cells.
    */
    void _addDependencyRuntimeState( DependencyState::DependencyId id,
                                    const Dependency &d );

    /*
        Compute the covering input box for all *remaining* queries.

        Using _nextQueryToSolve as the starting index, this recomputes:
            _currentLb[x] = min originalLbs[q][x]
            _currentUb[x] = max originalUbs[q][x]
        over all q ≥ _nextQueryToSolve.

        This shrinking box is used to tighten the input bounds of the
        preprocessed query and guide dependency pruning.
    */
    void _computeCoveringBoxFromRemainingQueries();

    /*
        Apply a collection of DeepPoly-proposed tightenings to the
        preprocessed query.

        For each Tightening t in `tightenings`, this updates the relevant
        bound in _preprocessedQuery (LB or UB), counting how many bounds
        were actually improved. This mirrors Engine’s bound-application
        logic but operates on the analyzer’s internal preprocessed query.

        Returns:
            The number of bounds tightened.
    */
    unsigned _applyTighteningsToPreprocessedQuery( const List<Tightening> &tightenings );


    /*
        Check whether the new input box [lbNew, ubNew] is a subset of
        (i.e., tighter than) the previous box [lbOld, ubOld].

        Returns true iff for every dimension x:
            lbNew[x] >= lbOld[x]   and   ubNew[x] <= ubOld[x]
        meaning the domain only shrank (never expanded).
    */
    bool _isSubset( const Vector<double> &lbNew,
                   const Vector<double> &ubNew,
                   const Vector<double> &lbOld,
                   const Vector<double> &ubOld ) const;

    /*
      Scan all WEIGHTED_SUM layers in the NLR, collect indices of unstable
      neurons, and store their variable IDs in _unstableNeurons.
    */
    void _collectAllUnstableNeurons();
    
    /*
      Check whether a given variable id belongs to the unstable set.
    */
    bool _isUnstableVar( unsigned var ) const;



};

#endif // __DependencyAnalyzer_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
