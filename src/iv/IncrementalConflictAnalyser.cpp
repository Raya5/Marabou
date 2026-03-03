#include "IncrementalConflictAnalyser.h"
#include "Query.h"
#include "FloatUtils.h"
#include <cmath> 

#include <cassert>

IncrementalConflictAnalyser::IncrementalConflictAnalyser()
    : _preprocessor( nullptr )
    , _currentEpsilon( -1.0 )
    , _currentQueryId( 0 )
    , _recordedConflictsForCurrent( true )
    , _queryIdWasSet( false )
    , _ancestorIds()
    , _ancestorsWasSet( false )
    , _clearBetweenRuns( true )
    , _autoInheritance( false )
    , _cadical( nullptr )
    , _conflictsExistForCurrent( false )
    , _relevantConflictsImported( false )
    , _bitmaskSize( 0 )
    , _threshold( 0 )
    , _recordedConflicts( 0 )
{
    if ( _autoInheritance || !_clearBetweenRuns )
    {
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Currently only ancestry-based inheritance with clearBetweenRuns=true is supported." );
    }

    _satVarToReluIndexMap.append( int(INFINITY) ); // index 0 unused
    if ( !_autoInheritance ) ASSERT( _clearBetweenRuns );
    if ( !_clearBetweenRuns )
    {
        _relevantConflictsImported = true;
        ASSERT (_minimalConflictBitmasks.size() == 0);
        _initializeSatSolver();
    }
}

IncrementalConflictAnalyser::~IncrementalConflictAnalyser()
{}

void IncrementalConflictAnalyser::setPreprocessor( Preprocessor *preprocessor )
{
    assert( preprocessor );
    _preprocessor = preprocessor;

    // Bitmask size will be initialized once SAT vars are known
}

void IncrementalConflictAnalyser::syncWithEngineBoundManager( BoundManager *boundManager )
{
    if ( !_conflictsExistForCurrent )
        return;
    ASSERT( _preprocessor );

    // If no mapping yet, nothing to seed
    if ( _reluIndexToSatVarMap.empty() )
        return;
    
    for ( const auto &it : _reluIndexToSatVarMap )
    {
        const unsigned oldVar = it.first;

        // old -> engine new index
        const unsigned newVar = _preprocessor->getNewIndex( oldVar );

        const double lb = boundManager->getLowerBound( newVar );
        const double ub = boundManager->getUpperBound( newVar );

        ASSERT( _getReluPhase( oldVar ) == ReLURuntimeState::Unstable );


        // If interval entirely >= 0 => Active
        if ( ( FloatUtils::areEqual( lb, 0.0 ) && FloatUtils::areEqual( ub, 0.0 ) ) || FloatUtils::gt( lb, ub ) )
        {

            _notifyLowerBoundUpdate(newVar, -INFINITY, lb);
            _notifyUpperBoundUpdate(newVar, +INFINITY, ub);
        }
        else if ( !FloatUtils::gt( ub, 0.0 ) )
        {
            _notifyUpperBoundUpdate(newVar, +INFINITY, ub);
        }
        else if ( !FloatUtils::lt( lb, 0.0 ) )
        {

            _notifyLowerBoundUpdate(newVar, -INFINITY, lb);
        }
        else
        {
            ASSERT( FloatUtils::lt( lb, 0.0 ) && FloatUtils::gt( ub, 0.0 ) ); // nothing to do
        }
    }
}

void IncrementalConflictAnalyser::setNewEpsilon( double epsilon )
{
    _currentEpsilon = epsilon;
}

void IncrementalConflictAnalyser::setID( unsigned id )
{
    // setID is only meaningful in ancestry mode
    ASSERT( !_autoInheritance );
    ASSERT( id > 0 );

    _currentQueryId = id;
    _queryIdWasSet = true;
}

void IncrementalConflictAnalyser::setRecordConflicts( bool recordConflicts )
{
    _recordedConflictsForCurrent = recordConflicts;
}

bool IncrementalConflictAnalyser::getRecordConflicts() const
{
    return _recordedConflictsForCurrent;
}

void IncrementalConflictAnalyser::setAncestors(
    const std::vector<unsigned> &ancestors )
{
    _ancestorIds = ancestors;
    _ancestorsWasSet = true;
}

bool IncrementalConflictAnalyser::addConflict(
    const std::vector<unsigned> &vars,
    const std::vector<bool> &isActiveList )
{
    ASSERT( vars.size() == isActiveList.size() );
    if ( vars.size() > _threshold )
    {
        return false;
    }
    if ( !_recordedConflictsForCurrent )
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Attempted to add conflict while recording disabled." );
    // Strategy-specific metadata must be set
    if ( _autoInheritance )
    {
        // Epsilon mode
        ASSERT( _currentEpsilon >= 0.0 );
    }
    else
    {
        if ( !_relevantConflictsImported ){
        ASSERT ( _clearBetweenRuns );
        _initializeSatSolver();
        _importRelevantConflicts();
        _relevantConflictsImported = true;
        }
        // Ancestry mode
        ASSERT( !_autoInheritance ); // todo remove
        ASSERT( _queryIdWasSet );
        ASSERT( _currentQueryId != 0 );
        ASSERT( _ancestorsWasSet );
    }

    // Minimality pruning (optional; keep if it works)
    IncrementalConflictAnalyser::Bitmask subMask =
        _buildConflictSubBitmask( vars, isActiveList );
    if ( _isNonMinimalConflict( subMask ) )
        return false;

    // Store conflict under the appropriate provenance key

    if ( _autoInheritance )
    {
        auto &bucket = _conflictsByEpsilon[_currentEpsilon];
        bucket.emplace_back( vars, isActiveList );
    }
    else
    {
        auto &bucket = _conflictsByQueryId[_currentQueryId];
        bucket.emplace_back( vars, isActiveList );
    }
    _recordedConflicts++;
    // Encode immediately into current SAT solver (so future calls in this run see it)

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned oldVar = vars[i];
        _reluIndexToSatVarForce( oldVar );
    }

    // Track minimality bitmask for this run
    IncrementalConflictAnalyser::Bitmask fullMask =
        _buildConflictBitmask( vars, isActiveList );
    _minimalConflictBitmasks.push_back( fullMask );
    return true;
}



void IncrementalConflictAnalyser::_notifyNeuronFixed( unsigned newVar, ReLUState state )
{
    ASSERT( _preprocessor );

    // engine new -> old var id
    const unsigned oldVar = _preprocessor->getOldIndex( newVar );

    ReLURuntimeState incoming =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Active : ReLURuntimeState::Inactive;

    const ReLURuntimeState opposite =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Inactive : ReLURuntimeState::Active;

    // If both phases were seen across time, treat as Zero
    auto it = _currentPhases.find( oldVar );
    if ( it != _currentPhases.end() && ( *it ).second == opposite )
        incoming = ReLURuntimeState::Zero;

    // If present, must be the opposite (otherwise inconsistent)
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

    // Crossed 0 from below => neuron guaranteed Active
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

    // Crossed 0 from above => neuron guaranteed Inactive
    if ( previousUpperBound > 0.0 && newUpperBound <= 0.0 )
        _notifyNeuronFixed( newVar, ReLUState::Inactive );
}


bool IncrementalConflictAnalyser::getImpliedTighteningsFromSat( List<Tightening> &tightenings )
{
    if ( !_relevantConflictsImported ){
        ASSERT ( _clearBetweenRuns );
        _initializeSatSolver();
        _importRelevantConflicts();
        _relevantConflictsImported = true;
    }
    bool noConflict;
    if ( !_conflictsExistForCurrent )
        noConflict = true;
    else
    {
        ASSERT( _currentEpsilon >= 0.0 or !_autoInheritance );
        ASSERT( _preprocessor );
        
        // Add assumptions for all currently fixed ReLUs in our SAT mapping.
        // IMPORTANT: ICA stores phases keyed by OLD vars, and the SAT mapping keys are OLD vars.
        bool someAssumed = false;

        for ( const auto &entry : _reluIndexToSatVarMap )
        {   
            const unsigned oldVar = entry.first;
            const ReLURuntimeState rt = _getReluPhase( oldVar );

            if ( rt == ReLURuntimeState::Active ){
                _cadical->assume( _phaseToLit( oldVar, ReLUState::Active ) );
                someAssumed = true;
            }
            else if ( rt == ReLURuntimeState::Inactive ){
                _cadical->assume( _phaseToLit( oldVar, ReLUState::Inactive ) );
                someAssumed = true;
            }
            // Zero/Unstable/Unseen: no assumption
        }

        if ( !someAssumed )
        {
            // No assumptions to propagate
            noConflict = true;

        }
        else
        {
            // Propagate under assumptions
            const int res = _cadical->propagate();

            // 20 means conflict under assumptions
            if ( res == 20 )
            {
                noConflict = false;
            }
            else
            {
                // Query entailed literals (implications) — same as DA
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
                    {
                        continue;
                    }

                    // Contradiction: SAT implies opposite of runtime-fixed value
                    if ( ( currentRt == ReLURuntimeState::Active   && impliedPhase == ReLUState::Inactive ) ||
                        ( currentRt == ReLURuntimeState::Inactive && impliedPhase == ReLUState::Active ) )
                    {
                        ASSERT( false );
                    }

                    // Zero treated as neither active nor inactive; do not force further
                    if ( currentRt == ReLURuntimeState::Zero )
                    {
                        continue;
                    }
                    // Emit tightening in ENGINE var space (NEW var id)
                    _emitTighteningsForImpliedPhase( oldVar, impliedPhase, tightenings );

                }
                noConflict = true;
            }
        }
    }
    _currentPhases.clear();
    return noConflict;
}


ReLURuntimeState IncrementalConflictAnalyser::_getReluPhase( unsigned oldVar ) const
{

    auto it = _currentPhases.find( oldVar );
    if ( it == _currentPhases.end() )
        return ReLURuntimeState::Unstable; // add this enum if you don't have it

    return ( *it ).second;
}

int IncrementalConflictAnalyser::_phaseToLit( unsigned oldVar, ReLUState phase ) const
{
    const unsigned satVar = _reluIndexToSatVar( oldVar );
    ASSERT( satVar != 0 );

    if ( phase == ReLUState::Active )
        return (int)satVar;   // x = true  -> Active
    else if ( phase == ReLUState::Inactive )
        return -(int)satVar;  // x = false -> Inactive

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
    {
        tightenings.append( Tightening( newVar, 0.0, Tightening::LB ) );
    }
    else if ( impliedPhase == ReLUState::Inactive )
    {
        tightenings.append( Tightening( newVar, 0.0, Tightening::UB ) );
    }
    else
    {
        ASSERT( false );
    }
}

void IncrementalConflictAnalyser::notifySolvingStarted( unsigned numQueryVariables )
{
    ASSERT( _preprocessor );
    ASSERT( !_conflictsExistForCurrent );

    // Epsilon must be valid
    if ( _autoInheritance )
    {
        ASSERT( _currentEpsilon >= 0.0 );
    }
    else
    {

        ASSERT( !_autoInheritance ); // TODO: remove later
        ASSERT( _clearBetweenRuns ); // ancestry requires rebuild-per-run
        ASSERT( _queryIdWasSet );
        ASSERT( _ancestorsWasSet );
    }
    if (_bitmaskSize == 0)
    {
        _bitmaskSize = numQueryVariables + 1;
        _threshold = std::max<std::size_t>(5, static_cast<std::size_t>(_bitmaskSize * 0.01));
    }

    // Bitmask must be large enough to index all variables
    ASSERT( _bitmaskSize >= numQueryVariables + 1 );

    if ( _clearBetweenRuns && _relevantConflictsImported)
    {
        throw MarabouError( MarabouError::DEBUGGING_ERROR,
                            "Conflicts already imported for this run." );
    }
}


void IncrementalConflictAnalyser::notifySolved()
{
    if ( _autoInheritance )
    {
        _currentEpsilon = -1.0;

        ASSERT( _currentQueryId == 0 );
        ASSERT( !_queryIdWasSet );
        ASSERT( _ancestorIds.empty() );
        ASSERT( !_ancestorsWasSet );
    }
    else
    {
        _currentQueryId = 0;
        _queryIdWasSet = false;
        _ancestorIds.clear();
        _ancestorsWasSet = false;

        ASSERT( _currentEpsilon == -1.0 );
    }
    _preprocessor = nullptr;
    _conflictsExistForCurrent = false;
    
    if ( _clearBetweenRuns )
    {
        _minimalConflictBitmasks.clear();
        _relevantConflictsImported = false;
    }
}

void IncrementalConflictAnalyser::_initializeSatSolver()
{
    _cadical = std::make_unique<CaDiCaL::Solver>();
    _cadical->declare_more_variables( _reluIndexToSatVarMap.size() + 1);
}

bool IncrementalConflictAnalyser::_isNonMinimalConflict(
    const IncrementalConflictAnalyser::Bitmask &mask ) const
{
    for ( const auto &known : _minimalConflictBitmasks )
    {
        if ( ( known & mask ) == known )
            return true;
    }
    return false;
}

IncrementalConflictAnalyser::Bitmask IncrementalConflictAnalyser::_buildConflictBitmask( const std::vector<unsigned> &vars,
                                                   const std::vector<bool> &isActive ) const
{
    ASSERT( vars.size() == isActive.size() );
    IncrementalConflictAnalyser::Bitmask mask( _bitmaskSize );

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

IncrementalConflictAnalyser::Bitmask IncrementalConflictAnalyser::_buildConflictSubBitmask( const std::vector<unsigned> &vars,
                                                      const std::vector<bool> &isActive ) const
{
    ASSERT( vars.size() == isActive.size() );
    IncrementalConflictAnalyser::Bitmask mask( _bitmaskSize );

    for ( unsigned i = 0; i < vars.size(); ++i )
    {
        const unsigned oldVar = vars[i];
        const bool active = isActive[i];

        const unsigned satVar = _reluIndexToSatVar( oldVar );
        if ( satVar == 0 )
            continue; // “sub” version skips unmapped

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
    // SAT vars start at 1; index 0 is reserved in _satVarToReluIndex
    unsigned newSatVar = _satVarToReluIndexMap.size();
    if ( !_clearBetweenRuns )
         _cadical->declare_one_more_variable();

    ASSERT( newSatVar > 0 );

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

        // Create SAT var if missing
        const unsigned satVar = _reluIndexToSatVar( oldVar );

        ASSERT( satVar > 0 );

        // Block the conflicting literal
        // If conflict says "active", we add ¬x; if "inactive", add x
        const int lit = active ? -(int)satVar : (int)satVar;
        if ( encodeNow )
            _cadical->add( lit );
    }

    // Terminate clause
    if ( encodeNow ) {
        _cadical->add( 0 );
        _conflictsExistForCurrent = true;
    }

}

unsigned IncrementalConflictAnalyser::_litBitIndex( unsigned satVar, bool isActive ) const
{
    // satVar starts at 1
    // bit 2*satVar     := Active literal  (x = true)
    // bit 2*satVar + 1 := Inactive literal (x = false)
    ASSERT( satVar > 0 );
    const unsigned base = 2 * satVar;
    return isActive ? base : base + 1;
}



void IncrementalConflictAnalyser::_importRelevantConflicts()
{
    if ( _autoInheritance )
        _importRelevantConflictsEpsilon();
    else
        _importRelevantConflictsAncestry();
}



void IncrementalConflictAnalyser::_importRelevantConflictsEpsilon()
{
    // size_t total = 0;
    // for ( const auto &kv : _conflictsByEpsilon )
    //     total += kv.second.size();

    unsigned imported = 0;

    for ( auto it = _conflictsByEpsilon.lower_bound( _currentEpsilon );
          it != _conflictsByEpsilon.end(); ++it )
    {
        for ( const Conflict &conflict : it->second )
        {
            _encodeConflictClause( conflict , true);
            ++imported;
        }
    }
    // printf(
    //     "[ICA][IV] _importRelevantConflictsEpsilon: currentEpsilon=%.6f, totalConflicts=%zu, imported=%u\n",
    //     _currentEpsilon,
    //     total,
    //     imported );
}

void IncrementalConflictAnalyser::_importRelevantConflictsAncestry()
{
    // Ancestry metadata must have been provided
    ASSERT( _ancestorsWasSet );
    ASSERT( _queryIdWasSet );
    ASSERT( _currentQueryId != 0 );

    // size_t total = 0;
    // for ( unsigned ancestorId : _ancestorIds )
    // {
    //     auto it = _conflictsByQueryId.find( ancestorId );
    //     if ( it != _conflictsByQueryId.end() )
    //         total += it->second.size();
    // }

    // printf(
    //     "[ICA][IV] _importRelevantConflictsAncestry: currentQueryId=%u, numAncestors=%zu, totalConflicts=%zu\n",
    //     _currentQueryId,
    //     _ancestorIds.size(),
    //     total );

    unsigned imported = 0;

    for ( unsigned ancestorId : _ancestorIds )
    {
        auto it = _conflictsByQueryId.find( ancestorId );
        if ( it == _conflictsByQueryId.end() )
            continue;

        for ( const Conflict &conflict : it->second )
        {
            _encodeConflictClause( conflict , true);
            ++imported;
        }
    }

    // printf(
    //     "[ICA][IV] _importRelevantConflictsAncestry done: imported=%u\n",
    //     imported );
}

unsigned IncrementalConflictAnalyser::getRecordedConflictCount() const
{
    return _recordedConflicts;
}
