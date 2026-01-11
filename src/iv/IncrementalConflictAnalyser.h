#ifndef __IncrementalConflictAnalyser_h__
#define __IncrementalConflictAnalyser_h__

#include <vector>
#include <unordered_map>

#include "Conflict.h"
#include "Preprocessor.h"

#include "cadical.hpp"
#include <boost/dynamic_bitset.hpp>

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

enum class ReLURuntimeState : uint8_t { Unstable, Active, Inactive, Zero };
enum class ReLUState : uint8_t { Active, Inactive };

class IncrementalConflictAnalyser
{
public:
    /*
      TODO
    */  
    typedef boost::dynamic_bitset<> Bitmask;

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
    void addConflict( const std::vector<unsigned> &vars,
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
    void notifySolvingStarted( unsigned numQueryVariables );

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
    bool _isNonMinimalConflict( const IncrementalConflictAnalyser::Bitmask &mask ) const;

    IncrementalConflictAnalyser::Bitmask _buildConflictBitmask(
        const std::vector<unsigned> &vars,
        const std::vector<bool> &isActive ) const;

    IncrementalConflictAnalyser::Bitmask _buildConflictSubBitmask(
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
    std::vector<Bitmask> _minimalConflictBitmasks;
    unsigned _bitmaskSize;

    /*
     * Phase tracking (context-dependent)
     */
    CVC4::context::CDHashMap<unsigned, ReLURuntimeState, std::hash<unsigned>> *_seenPhase;
};

#endif // __IncrementalConflictAnalyser_h__
