#ifndef __IncrementalConflictAnalyser_h__
#define __IncrementalConflictAnalyser_h__

#include <vector>
#include <unordered_map>

#include "Conflict.h"
#include "DependencyAnalyzer.h"   // for Bitmask, ReLURuntimeState, etc.
#include "Preprocessor.h"
#include "Statistics.h"

#include "cadical.hpp"

#include "context/context.h"
#include "context/cdhashmap.h"
class Query;

/*
 * IncrementalConflictAnalyser
 * ---------------------------
 *
 * A lightweight replacement for DependencyAnalyzer, responsible ONLY for:
 *   - collecting SAT-style conflicts learned during solving
 *   - storing them with the epsilon they were learned under
 *   - reusing them across solves (ONLY_LAST or ALL_LAST)
 *   - running SAT-based reasoning to derive implied tightenings (stub for now)
 *
 * No dependency discovery, no NLR logic, no preprocessing passes.
 */
class IncrementalConflictAnalyser
{
public:
    /*
     * Constructor
     *
     * reuseAllConflicts:
     *   false -> ONLY_LAST  (keep CaDiCaL state across solves)
     *   true  -> ALL_LAST   (reset CaDiCaL per epsilon and re-import conflicts)
     */
    explicit IncrementalConflictAnalyser( bool reuseAllConflicts );

    ~IncrementalConflictAnalyser();

    /*
     * Lifecycle hooks (mirrors DA)
     */
    void setContext( CVC4::context::Context *context );
    void setPreprocessor( Preprocessor *preprocessor );

    /* 
     * Sync already-tightened bounds from Engine and seed _seenPhase for stable ReLUs
     */
    void syncWithEnginePreprocessedQuery( const Query &engineQuery );


    /*
     * Called once per epsilon solve, before solving begins
     */
    void setNewEpsilon( double epsilon );

    /*
     * Conflict ingestion (called by Engine)
     *
     * vars / isActiveList:
     *   parallel vectors describing a phase conflict
     */
    bool addConflict( const std::vector<unsigned> &vars,
                      const std::vector<bool> &isActiveList );

    /*
     * Notifications from Engine (stubs for now)
     */
    void notifyNeuronFixed( unsigned newVar, ReLUState state );
    void notifyLowerBoundUpdate( unsigned newVar,
                             double previousLowerBound,
                             double newLowerBound );
    void notifyUpperBoundUpdate( unsigned newVar,
                             double previousUpperBound,
                             double newUpperBound );

    /*
     * Query implied tightenings (stub)
     *
     * Returns false for now (no tightenings)
     */
    bool getImpliedTighteningsFromSat( List<Tightening> &tightenings );

    /*
    * Called once per epsilon solve, right before solving begins.
    * Used for sanity checks and final initialization.
    */
    void notifySolvingStarted( unsigned numQueryVariables,
                            const Query *engineQuery,
                            NLR::NetworkLevelReasoner *nlr,
                            BoundManager *boundManager,
                            Statistics *statistics );

    void setEngineQuery( const Query *q );
    void setNetworkLevelReasoner( NLR::NetworkLevelReasoner *nlr );
    void setBoundManager( BoundManager *bm );
    void setStatistics( Statistics *statistics );

    bool isNonMinimalDependencyVarsSubMask(const std::vector<unsigned> &vars ) const;

    /*
    Add a confirmed dependency (vars + forbidden phases) and store it as a conflict.

    - Performs early minimality pruning using VAR-ONLY masks (ignores polarity).
    - If kept, records the dependency var-mask so future supersets are pruned early.
    - Then calls addConflict(vars, isActiveList) which stores+encodes into CaDiCaL.

    `vars` are old-var indices, must be same length as `isActiveList`,
    and should be in canonical sorted order.
    */
    void addDependency( const std::vector<unsigned> &vars,
                        const std::vector<bool> &isActiveList );


    /*
     * End-of-solve notification (optional hook)
     */
    void notifySolved();

private:
    /*
     * Internal helpers
     */
    void _initializeSatSolver();
    void _importRelevantConflicts();
    bool _isNonMinimalConflict( const DependencyAnalyzer::Bitmask &mask ) const;

    DependencyAnalyzer::Bitmask _buildConflictBitmask(
        const std::vector<unsigned> &vars,
        const std::vector<bool> &isActive ) const;

    DependencyAnalyzer::Bitmask _buildConflictSubBitmask(
        const std::vector<unsigned> &vars,
        const std::vector<bool> &isActive ) const;
        

    unsigned _reluIndexToSatVar( unsigned relu ) const;
    unsigned _reluIndexToSatVarForce( unsigned relu );
    unsigned _createNewSatVarForRelu( unsigned relu ); 
    unsigned _satVarToReluIndex( unsigned satVar ) const;

    void _encodeConflictClause( const Conflict &conflict );

    int  _phaseToLit( unsigned oldVar, ReLUState phase ) const;
    bool _litToPhase( int lit, unsigned &oldVar, ReLUState &phase ) const;
    ReLURuntimeState _getReluPhase( unsigned oldVar ) const;

    unsigned _litBitIndex( unsigned satVar, bool isActive ) const;

    void _emitTighteningsForImpliedPhase( unsigned oldVar,
                                        ReLUState impliedPhase,
                                        List<Tightening> &tightenings ) const;


    
    /*
    Return true iff `mask` is a superset of a previously-recorded *minimal dependency var-set*.

    This is used by the DependencyCalculator *before* running expensive slicing / analysis,
    to avoid analyzing neuron sets whose var-set already contains a known smaller dependency.
    */
    bool _isNonMinimalDependencyVars( const DependencyAnalyzer::Bitmask &mask ) const;

    /*
    Build a full dependency-var bitmask from a set of ReLU vars (old indexing).
    Each variable corresponds to a single bit (no polarity).

    Requires that all vars already have a mapping oldVar -> satVar (i.e., satVar != 0),
    because the bit positions are SAT-var indexed.
    */
    DependencyAnalyzer::Bitmask
    _buildDependencyVarsBitmask( const std::vector<unsigned> &vars ) const;

    /*
    Build a *sub* dependency-var bitmask from a set of ReLU vars (old indexing).
    Variables that do not yet have a SAT mapping are skipped.

    This allows early pruning checks even if the SAT mapping is incomplete.
    */
    DependencyAnalyzer::Bitmask
    _buildDependencyVarsSubBitmask( const std::vector<unsigned> &vars ) const;

    /*
    Record a newly found dependency var-set as minimal:
    - store the full bitmask in `_minimalDependencyVarBitmasks`
    - intended to be called after we actually confirm a dependency exists
        (and we are about to add it as a conflict / nogood).
    */
    void _recordMinimalDependencyVars( const std::vector<unsigned> &vars );
    void _encodeMinimalBitmasks( const Conflict &conflict );

    void calculateDependencies();





private:
    /*
     * Context / preprocessing
     */
    CVC4::context::Context *_context;
    Preprocessor *_preprocessor;

    /*
     * Epsilon tracking
     */
    double _currentEpsilon;

    /*
     * Conflict storage (flat)
     */
    std::vector<Conflict> _conflicts;

    /*
     * Reuse policy
     *   false -> ONLY_LAST
     *   true  -> ALL_LAST
     */
    bool _reuseAllConflicts;

    /*
     * SAT-based reasoning
     */
    std::unique_ptr<CaDiCaL::Solver> _cadical;

    std::unordered_map<unsigned, unsigned> _reluIndexToSatVarMap;
    Vector<unsigned> _satVarToReluIndexMap;


    /*
     * Minimal conflict tracking (SAT-var indexed)
     */
    std::vector<DependencyAnalyzer::Bitmask> _minimalConflictBitmasks;

    // Store minimal dependency var-sets as bitmasks (vars only, no polarity)
    std::vector<DependencyAnalyzer::Bitmask> _minimalDependencyVarBitmasks;
    
    unsigned _bitmaskConflictSize;
    // Bitmask size for dependency-var minimality (set in notifySolvingStarted)
    unsigned _bitmaskDependencyVSize;


    /*
     * Phase tracking (context-dependent)
     */
    CVC4::context::CDHashMap<unsigned, ReLURuntimeState, std::hash<unsigned>> *_seenPhase;


    // IncrementalConflictAnalyser.h (private)
    const Query *_engineQuery;                 // current preprocessed query (Engine-owned)
    NLR::NetworkLevelReasoner *_nlr;     // borrowed from _engineQuery
    BoundManager *_boundManager;               // optional sanity
    Statistics *_statistics;
    
    // Stats collected from DependencyCalculator
    unsigned _ws1_unstable = 0, _ws1_depsFound = 0;
    double _ws1_seconds = 0.0;

    unsigned _ws2_unstable = 0, _ws2_depsFound = 0;
    double _ws2_seconds = 0.0;


};

#endif // __IncrementalConflictAnalyser_h__
