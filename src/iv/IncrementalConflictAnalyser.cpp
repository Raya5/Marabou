#include "IncrementalConflictAnalyser.h"

#include "Query.h"
#include "FloatUtils.h"

#include <cassert>
#include <cmath>
#include <algorithm>

IncrementalConflictAnalyser::IncrementalConflictAnalyser()
    : _preprocessor( nullptr )
    , _currentQueryId( 0 )
    , _recordedConflictsForCurrent( true )
    , _queryIdWasSet( false )
    , _ancestorIds()
    , _ancestorsWasSet( false )
    , _conflictsByQueryId()
    , _cadical( nullptr )
    , _conflictsExistForCurrent( false )
    , _relevantConflictsImported( false )
    , _reluIndexToSatVarMap()
    , _satVarToReluIndexMap()
    , _minimalConflictBitmasks()
    , _bitmaskSize( 0 )
    , _threshold( 0 )
    , _currentPhases()
    , _recordedConflicts( 0 )
{
    // SAT vars start at 1; index 0 is unused
    _satVarToReluIndexMap.append( (unsigned)INFINITY );
}

IncrementalConflictAnalyser::~IncrementalConflictAnalyser()
{}

void IncrementalConflictAnalyser::setPreprocessor( Preprocessor *preprocessor )
{
    assert( preprocessor );
    _preprocessor = preprocessor;
}

void IncrementalConflictAnalyser::syncWithEngineBoundManager( BoundManager *boundManager )
{
    if ( !_conflictsExistForCurrent )
        return;

    ASSERT( _preprocessor );

    if ( _reluIndexToSatVarMap.empty() )
        return;

    for ( const auto &it : _reluIndexToSatVarMap )
    {
        const unsigned oldVar = it.first;
        const unsigned newVar = _preprocessor->getNewIndex( oldVar );

        const double lb = boundManager->getLowerBound( newVar );
        const double ub = boundManager->getUpperBound( newVar );

        ASSERT( _getReluPhase( oldVar ) == ReLURuntimeState::Unstable );

        // If interval entirely <= 0 => Inactive
        if ( !FloatUtils::gt( ub, 0.0 ) )
        {
            _notifyUpperBoundUpdate( newVar, +INFINITY, ub );
        }
        // If interval entirely >= 0 => Active
        else if ( !FloatUtils::lt( lb, 0.0 ) )
        {
            _notifyLowerBoundUpdate( newVar, -INFINITY, lb );
        }
        else if ( FloatUtils::gt( lb, ub ) )
        {
            // Infeasible: treat as both active and inactive to trigger conflict
            _notifyLowerBoundUpdate( newVar, -INFINITY, lb );
            _notifyUpperBoundUpdate( newVar, +INFINITY, ub );
        }
        else
        {
            // Still unstable: lb < 0 < ub
            ASSERT( FloatUtils::lt( lb, 0.0 ) && FloatUtils::gt( ub, 0.0 ) );
        }
    }
}

void IncrementalConflictAnalyser::setID( unsigned id )
{
    ASSERT( id > 0 );
    _currentQueryId = id;
    _queryIdWasSet = true;
}

void IncrementalConflictAnalyser::setAncestors( const std::vector<unsigned> &ancestors )
{
    _ancestorIds = ancestors;
    _ancestorsWasSet = true;
}

void IncrementalConflictAnalyser::setRecordConflicts( bool recordConflicts )
{
    _recordedConflictsForCurrent = recordConflicts;
}

bool IncrementalConflictAnalyser::getRecordConflicts() const
{
    return _recordedConflictsForCurrent;
}

void IncrementalConflictAnalyser::notifySolvingStarted( unsigned numQueryVariables )
{
    ASSERT( _preprocessor );
    ASSERT( _queryIdWasSet );
    ASSERT( _ancestorsWasSet );
    ASSERT( !_relevantConflictsImported );
    ASSERT( !_cadical ); // always rebuild per solve/run

    // Bitmask sizing (one-time initialization; safe to keep stable across runs)
    if ( _bitmaskSize == 0 )
    {
        _bitmaskSize = numQueryVariables + 1;
        _threshold = std::max<std::size_t>(
            5,
            static_cast<std::size_t>( _bitmaskSize * 0.01 ) );
    }

    if ( _bitmaskSize < numQueryVariables ){
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Number of query variables exceeds ICA bitmask capacity." );
    }

    // Run-local state
    _minimalConflictBitmasks.clear();
    _currentPhases.clear();
}

void IncrementalConflictAnalyser::notifySolved()
{
    // Clear run metadata
    _currentQueryId = 0;
    _queryIdWasSet = false;
    _ancestorIds.clear();
    _ancestorsWasSet = false;

    _preprocessor = nullptr;

    _conflictsExistForCurrent = false;
    _relevantConflictsImported = false;

    // Rebuilt every run, so drop solver + run-local minimality cache
    _cadical.reset();
    _minimalConflictBitmasks.clear();
    _currentPhases.clear();
}

unsigned IncrementalConflictAnalyser::getRecordedConflictCount() const
{
    return _recordedConflicts;
}

bool IncrementalConflictAnalyser::addConflict( const std::vector<unsigned> &vars,
                                               const std::vector<bool> &isActiveList )
{
    ASSERT( vars.size() == isActiveList.size() );

    // Threshold-based filter (size cutoff)
    if ( vars.size() > _threshold )
        return false;

    if ( !_recordedConflictsForCurrent )
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Attempted to add conflict while recording disabled." );

    // Ancestry metadata must be set for this run
    ASSERT( _queryIdWasSet );
    ASSERT( _ancestorsWasSet );
    ASSERT( _currentQueryId != 0 );
    ASSERT( _bitmaskSize > 0 );

    // Ensure SAT solver exists + import ancestor conflicts once per run
    if ( !_relevantConflictsImported )
    {
        _initializeSatSolver();
        _importRelevantConflicts();
        _relevantConflictsImported = true;
    }

    // Minimality pruning (sub-mask skips currently unmapped vars)
    Bitmask subMask = _buildConflictSubBitmask( vars, isActiveList );
    if ( _isNonMinimalConflict( subMask ) )
        return false;

    // Store conflict under current query ID
    auto &bucket = _conflictsByQueryId[_currentQueryId];
    bucket.emplace_back( vars, isActiveList );

    ++_recordedConflicts;

    // Ensure SAT vars exist
    for ( unsigned i = 0; i < vars.size(); ++i )
        _reluIndexToSatVarForce( vars[i] );

    // Track minimality bitmask (full, SAT-mapped)
    Bitmask fullMask = _buildConflictBitmask( vars, isActiveList );
    _minimalConflictBitmasks.push_back( fullMask );

    // Encode immediately so propagation in this same run can use it
    _encodeConflictClause( bucket.back(), /*encodeNow=*/true );

    return true;
}

bool IncrementalConflictAnalyser::getImpliedTighteningsFromSat( List<Tightening> &tightenings )
{
    // Ensure SAT solver exists + ancestors imported
    if ( !_relevantConflictsImported )
    {
        _initializeSatSolver();
        _importRelevantConflicts();
        _relevantConflictsImported = true;
    }

    bool noConflict = true;

    if ( _conflictsExistForCurrent )
    {
        ASSERT( _preprocessor );

        // Add assumptions for all currently fixed phases in SAT mapping
        bool someAssumed = false;
        for ( const auto &entry : _reluIndexToSatVarMap )
        {
            const unsigned oldVar = entry.first;
            const ReLURuntimeState rt = _getReluPhase( oldVar );

            if ( rt == ReLURuntimeState::Active )
            {
                _cadical->assume( _phaseToLit( oldVar, ReLUState::Active ) );
                someAssumed = true;
            }
            else if ( rt == ReLURuntimeState::Inactive )
            {
                _cadical->assume( _phaseToLit( oldVar, ReLUState::Inactive ) );
                someAssumed = true;
            }
        }

        if ( someAssumed )
        {
            const int res = _cadical->propagate();

            // 20 means conflict under assumptions
            if ( res == 20 )
            {
                noConflict = false;
            }
            else
            {
                std::vector<int> implicants;
                _cadical->implied( implicants );

                for ( int lit : implicants )
                {
                    unsigned oldVar = 0;
                    ReLUState impliedPhase;

                    const bool success = _litToPhase( lit, oldVar, impliedPhase );
                    if ( !success )
                        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                                            "Failed to decode implied literal from SAT solver." );

                    const ReLURuntimeState currentRt = _getReluPhase( oldVar );

                    // Skip if already fixed consistently
                    if ( ( currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Active ) ||
                         ( currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Inactive ) )
                        continue;

                    // Contradiction: SAT implies opposite of runtime-fixed value
                    if ( ( currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Inactive ) ||
                         ( currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Active ) )
                        ASSERT( false );

                    // Zero treated as neither; do not force further
                    if ( currentRt == ReLURuntimeState::Zero )
                        continue;

                    _emitTighteningsForImpliedPhase( oldVar, impliedPhase, tightenings );
                }

                noConflict = true;
            }
        }
    }

    // Clear runtime phases after producing tightenings for this call
    _currentPhases.clear();
    return noConflict;
}

// ----------------------------------------------------------------------
// Internal helpers
// ----------------------------------------------------------------------

void IncrementalConflictAnalyser::_initializeSatSolver()
{
    _cadical = std::make_unique<CaDiCaL::Solver>();

    // Declare enough variables for current mapping (plus index-0 dummy)
    _cadical->declare_more_variables( _satVarToReluIndexMap.size() );
}

void IncrementalConflictAnalyser::_importRelevantConflicts()
{
    _importRelevantConflictsAncestry();
}

void IncrementalConflictAnalyser::_importRelevantConflictsAncestry()
{
    ASSERT( _ancestorsWasSet );
    ASSERT( _queryIdWasSet );
    ASSERT( _currentQueryId != 0 );
    ASSERT( _cadical );

    for ( unsigned ancestorId : _ancestorIds )
    {
        auto it = _conflictsByQueryId.find( ancestorId );
        if ( it == _conflictsByQueryId.end() )
            continue;

        for ( const Conflict &conflict : it->second )
        {
            // Ensure SAT vars exist for all vars in the imported conflict
            const auto &vars = conflict.getVars();
            for ( unsigned v : vars )
                _reluIndexToSatVarForce( v );

            _encodeConflictClause( conflict, /*encodeNow=*/true );
        }
    }
}

bool IncrementalConflictAnalyser::_isNonMinimalConflict( const Bitmask &mask ) const
{
    for ( const auto &known : _minimalConflictBitmasks )
    {
        if ( ( known & mask ) == known )
            return true;
    }
    return false;
}

IncrementalConflictAnalyser::Bitmask
IncrementalConflictAnalyser::_buildConflictBitmask( const std::vector<unsigned> &vars,
                                                    const std::vector<bool> &isActive ) const
{
    ASSERT( vars.size() == isActive.size() );
    Bitmask mask( _bitmaskSize );

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned oldVar = vars[i];
        const bool active = isActive[i];

        const unsigned satVar = _reluIndexToSatVar( oldVar );
        ASSERT( satVar != 0 );

        const unsigned bit = _litBitIndex( satVar, active );
        ASSERT( bit < mask.size() );
        mask.set( bit );
    }

    return mask;
}

IncrementalConflictAnalyser::Bitmask
IncrementalConflictAnalyser::_buildConflictSubBitmask( const std::vector<unsigned> &vars,
                                                       const std::vector<bool> &isActive ) const
{
    ASSERT( vars.size() == isActive.size() );
    Bitmask mask( _bitmaskSize );

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned oldVar = vars[i];
        const bool active = isActive[i];

        const unsigned satVar = _reluIndexToSatVar( oldVar );
        if ( satVar == 0 )
            continue; // skip unmapped in sub-mask

        const unsigned bit = _litBitIndex( satVar, active );
        ASSERT( bit < mask.size() );
        mask.set( bit );
    }

    return mask;
}

unsigned IncrementalConflictAnalyser::_reluIndexToSatVar( unsigned relu ) const
{
    auto it = _reluIndexToSatVarMap.find( relu );
    if ( it == _reluIndexToSatVarMap.end() )
        return 0;
    return it->second;
}

unsigned IncrementalConflictAnalyser::_reluIndexToSatVarForce( unsigned relu )
{
    const unsigned existing = _reluIndexToSatVar( relu );
    if ( existing != 0 )
        return existing;

    return _createNewSatVarForRelu( relu );
}

unsigned IncrementalConflictAnalyser::_createNewSatVarForRelu( unsigned relu )
{
    // SAT vars start at 1; index 0 reserved
    unsigned newSatVar = _satVarToReluIndexMap.size();
    ASSERT( newSatVar > 0 );

    // If solver already exists this run, extend it
    if ( _cadical )
        _cadical->declare_one_more_variable();

    _reluIndexToSatVarMap[relu] = newSatVar;
    _satVarToReluIndexMap.append( relu );
    return newSatVar;
}

unsigned IncrementalConflictAnalyser::_satVarToReluIndex( unsigned satVar ) const
{
    ASSERT( satVar > 0 );
    ASSERT( satVar < _satVarToReluIndexMap.size() );
    return _satVarToReluIndexMap[satVar];
}

void IncrementalConflictAnalyser::_encodeConflictClause( const Conflict &conflict, bool encodeNow )
{
    ASSERT( _cadical );

    const auto &vars = conflict.getVars();
    const auto &isActive = conflict.getIsActive();

    ASSERT( vars.size() == isActive.size() );

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned oldVar = vars[i];
        const bool active = isActive[i];

        const unsigned satVar = _reluIndexToSatVar( oldVar );
        ASSERT( satVar > 0 );

        // Block the conflicting literal:
        // if conflict says "active", add ¬x; if "inactive", add x
        const int lit = active ? -(int)satVar : (int)satVar;
        if ( encodeNow )
            _cadical->add( lit );
    }

    if ( encodeNow )
    {
        _cadical->add( 0 ); // terminate clause
        _conflictsExistForCurrent = true;
    }
}

unsigned IncrementalConflictAnalyser::_litBitIndex( unsigned satVar, bool isActive ) const
{
    // bit 2*satVar     := Active literal  (x = true)
    // bit 2*satVar + 1 := Inactive literal (x = false)
    ASSERT( satVar > 0 );
    const unsigned base = 2 * satVar;
    return isActive ? base : base + 1;
}

ReLURuntimeState IncrementalConflictAnalyser::_getReluPhase( unsigned oldVar ) const
{
    auto it = _currentPhases.find( oldVar );
    if ( it == _currentPhases.end() )
        return ReLURuntimeState::Unstable;
    return ( *it ).second;
}

int IncrementalConflictAnalyser::_phaseToLit( unsigned oldVar, ReLUState phase ) const
{
    const unsigned satVar = _reluIndexToSatVar( oldVar );
    ASSERT( satVar != 0 );

    if ( phase == ReLUState::Active )
        return (int)satVar;    // true -> Active
    else if ( phase == ReLUState::Inactive )
        return -(int)satVar;   // false -> Inactive

    ASSERT( false );
    return 0;
}

bool IncrementalConflictAnalyser::_litToPhase( int lit,
                                              unsigned &oldVar,
                                              ReLUState &phase ) const
{
    if ( lit == 0 )
        return false;

    const unsigned satVar = (unsigned)std::abs( lit );
    if ( satVar == 0 || satVar >= _satVarToReluIndexMap.size() )
        return false;

    oldVar = _satVarToReluIndexMap[satVar];
    phase = ( lit > 0 ) ? ReLUState::Active : ReLUState::Inactive;
    return true;
}

void IncrementalConflictAnalyser::_emitTighteningsForImpliedPhase( unsigned oldVar,
                                                                  ReLUState impliedPhase,
                                                                  List<Tightening> &tightenings ) const
{
    ASSERT( _preprocessor );
    const unsigned newVar = _preprocessor->getNewIndex( oldVar );

    if ( impliedPhase == ReLUState::Active )
        tightenings.append( Tightening( newVar, 0.0, Tightening::LB ) );
    else if ( impliedPhase == ReLUState::Inactive )
        tightenings.append( Tightening( newVar, 0.0, Tightening::UB ) );
    else
        ASSERT( false );
}

// ----------------------------------------------------------------------
// Engine notifications -> phase tracking
// ----------------------------------------------------------------------

void IncrementalConflictAnalyser::_notifyNeuronFixed( unsigned newVar, ReLUState state )
{
    ASSERT( _preprocessor );

    const unsigned oldVar = _preprocessor->getOldIndex( newVar );

    ReLURuntimeState incoming =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Active : ReLURuntimeState::Inactive;

    const ReLURuntimeState opposite =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Inactive : ReLURuntimeState::Active;

    auto it = _currentPhases.find( oldVar );
    if ( it != _currentPhases.end() && ( *it ).second == opposite )
        incoming = ReLURuntimeState::Zero;

    ASSERT( it == _currentPhases.end() || ( *it ).second == opposite );

    _currentPhases[oldVar] = incoming;
}

void IncrementalConflictAnalyser::_notifyLowerBoundUpdate( unsigned newVar,
                                                          double previousLowerBound,
                                                          double newLowerBound )
{
    ASSERT( _preprocessor );

    // Lower bounds must only tighten upward
    if ( !FloatUtils::gt( newLowerBound, previousLowerBound ) )
        return;
    ASSERT( !FloatUtils::lt( newLowerBound, previousLowerBound ) );

    // Crossed 0 from below => Active
    if ( previousLowerBound < 0.0 && newLowerBound >= 0.0 )
        _notifyNeuronFixed( newVar, ReLUState::Active );
}

void IncrementalConflictAnalyser::_notifyUpperBoundUpdate( unsigned newVar,
                                                          double previousUpperBound,
                                                          double newUpperBound )
{
    ASSERT( _preprocessor );

    // Upper bounds must only tighten downward
    if ( !FloatUtils::lt( newUpperBound, previousUpperBound ) )
        return;
    ASSERT( !FloatUtils::gt( newUpperBound, previousUpperBound ) );

    // Crossed 0 from above => Inactive
    if ( previousUpperBound > 0.0 && newUpperBound <= 0.0 )
        _notifyNeuronFixed( newVar, ReLUState::Inactive );
}