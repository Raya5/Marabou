#ifndef __IncrementalConflictAnalyser_h__
#define __IncrementalConflictAnalyser_h__

#include <vector>
#include <unordered_map>
#include <memory>

#include "Conflict.h"
#include "Preprocessor.h"

#include "cadical.hpp"
#include <boost/dynamic_bitset.hpp>

class Query;

/*
 * IncrementalConflictAnalyser
 * ---------------------------
 *
 * An ancestry incremental conflict component, responsible for:
 *   - collecting SAT-style phase conflicts learned during solving
 *   - storing them by query ID
 *   - reusing them across solves using explicit ancestry metadata
 *   - running SAT-based reasoning to derive implied tightenings
 *
 * Notes:
 *   - Always rebuilds CaDiCaL per run (per solve invocation).
 *   - No dependency discovery / preprocessing passes beyond required mapping hooks.
 */

enum class ReLURuntimeState : uint8_t { Unstable, Active, Inactive, Zero };
enum class ReLUState : uint8_t { Active, Inactive };

class IncrementalConflictAnalyser
{
public:
    typedef boost::dynamic_bitset<> Bitmask;

    explicit IncrementalConflictAnalyser();
    ~IncrementalConflictAnalyser();

    void setPreprocessor( Preprocessor *preprocessor );

    /*
     * Sync already-tightened bounds from Engine and seed _currentPhases for stable ReLUs
     */
    void syncWithEngineBoundManager( BoundManager *boundManager );

    /*
     * Enable/disable conflict recording for the current query.
     * Engine must not call addConflict() when recording is disabled.
     */
    void setRecordConflicts( bool recordConflicts );
    bool getRecordConflicts() const;

    /*
     * Ancestry-based inheritance metadata:
     *   - set current query ID, must be > 0
     *   - set current query ancestors (IDs of prior queries whose conflicts may be reused)
     */
    void setID( unsigned id );
    void setAncestors( const std::vector<unsigned> &ancestors );

    /*
     * Conflict ingestion (called by Engine)
     *
     * vars / isActiveList:
     *   parallel vectors describing a phase conflict
     */
    bool addConflict( const std::vector<unsigned> &vars,
                      const std::vector<bool> &isActiveList );

    /*
     * Query implied tightenings via SAT propagation under current fixed phases.
     * Returns false if a conflict is detected under assumptions.
     */
    bool getImpliedTighteningsFromSat( List<Tightening> &tightenings );

    /*
     * Called once per solve, right before solving begins.
     * Used for sanity checks and final initialization.
     */
    void notifySolvingStarted( unsigned numQueryVariables );

    /*
     * End-of-solve notification.
     */
    void notifySolved();

    /*
     * Stats: total conflicts recorded over ICA lifetime
     */
    unsigned getRecordedConflictCount() const;

private:
    // SAT solver lifecycle / import
    void _initializeSatSolver();
    void _importRelevantConflicts();
    void _importRelevantConflictsAncestry();

    // Minimality pruning
    bool _isNonMinimalConflict( const Bitmask &mask ) const;

    Bitmask _buildConflictBitmask( const std::vector<unsigned> &vars,
                                   const std::vector<bool> &isActive ) const;

    Bitmask _buildConflictSubBitmask( const std::vector<unsigned> &vars,
                                      const std::vector<bool> &isActive ) const;

    // Notifications from Engine (phase tracking)
    void _notifyNeuronFixed( unsigned newVar, ReLUState state );
    void _notifyLowerBoundUpdate( unsigned newVar,
                                  double previousLowerBound,
                                  double newLowerBound );
    void _notifyUpperBoundUpdate( unsigned newVar,
                                  double previousUpperBound,
                                  double newUpperBound );

    // SAT var mapping
    unsigned _reluIndexToSatVar( unsigned relu ) const;
    unsigned _reluIndexToSatVarForce( unsigned relu );
    unsigned _createNewSatVarForRelu( unsigned relu );
    unsigned _satVarToReluIndex( unsigned satVar ) const;

    // Clause encoding / decoding
    void _encodeConflictClause( const Conflict &conflict, bool encodeNow );

    int  _phaseToLit( unsigned oldVar, ReLUState phase ) const;
    bool _litToPhase( int lit, unsigned &oldVar, ReLUState &phase ) const;
    ReLURuntimeState _getReluPhase( unsigned oldVar ) const;

    unsigned _litBitIndex( unsigned satVar, bool isActive ) const;

    // Emit implied tightenings in ENGINE var space (new indices)
    void _emitTighteningsForImpliedPhase( unsigned oldVar,
                                          ReLUState impliedPhase,
                                          List<Tightening> &tightenings ) const;

private:
    // Preprocessing / old<->new mapping
    Preprocessor *_preprocessor;

    // Query ID / ancestry tracking (must be set each run)
    unsigned _currentQueryId;
    bool _recordedConflictsForCurrent;
    bool _queryIdWasSet;
    std::vector<unsigned> _ancestorIds;
    bool _ancestorsWasSet;

    // Conflict storage (organized by query ID)
    std::unordered_map<unsigned, std::vector<Conflict>> _conflictsByQueryId;

    // SAT-based reasoning (rebuilt per run)
    std::unique_ptr<CaDiCaL::Solver> _cadical;
    bool _conflictsExistForCurrent;
    bool _relevantConflictsImported;

    // Mapping between old ReLU variable index and SAT variable
    std::unordered_map<unsigned, unsigned> _reluIndexToSatVarMap;
    Vector<unsigned> _satVarToReluIndexMap; // index 0 unused

    // Minimal conflict tracking (bitmask indexed)
    std::vector<Bitmask> _minimalConflictBitmasks;
    unsigned _bitmaskSize;
    std::size_t _threshold;

    // Runtime phase tracking (keyed by old vars)
    std::unordered_map<unsigned, ReLURuntimeState> _currentPhases;

    // Total conflicts recorded over ICA lifetime for stats
    unsigned _recordedConflicts;
};

#endif // __IncrementalConflictAnalyser_h__