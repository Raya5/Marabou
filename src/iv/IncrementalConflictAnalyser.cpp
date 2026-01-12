#include "IncrementalConflictAnalyser.h"
#include "Query.h"
#include "FloatUtils.h"
#include <cmath> 

#include <cassert>

IncrementalConflictAnalyser::IncrementalConflictAnalyser( bool reuseAllConflicts )
    : _context( nullptr )
    , _preprocessor( nullptr )
    , _currentEpsilon( -1.0 )
    , _reuseAllConflicts( reuseAllConflicts )
    , _cadical( nullptr )
    , _conflictsExistForCurrent( false )
    , _bitmaskSize( 0 )
    , _seenPhase( nullptr )
{
}

IncrementalConflictAnalyser::~IncrementalConflictAnalyser()
{}

void IncrementalConflictAnalyser::setContext( CVC4::context::Context *context )
{
    assert( context );
    _context = context;

    _seenPhase =
        new (true) CVC4::context::CDHashMap<unsigned, ReLURuntimeState, std::hash<unsigned>>( _context );
}

void IncrementalConflictAnalyser::setPreprocessor( Preprocessor *preprocessor )
{
    assert( preprocessor );
    _preprocessor = preprocessor;

    // Bitmask size will be initialized once SAT vars are known
}

void IncrementalConflictAnalyser::syncWithEnginePreprocessedQuery( const Query &engineQuery )
{
    ASSERT( _context );
    ASSERT( _preprocessor );
    ASSERT( _seenPhase );

    // If no mapping yet, nothing to seed
    if ( _reluIndexToSatVarMap.empty() )
        return;

    for ( const auto &it : _reluIndexToSatVarMap )
    {
        const unsigned oldVar = it.first;

        // old -> engine new index
        const unsigned newVar = _preprocessor->getNewIndex( oldVar );

        const double lb = engineQuery.getLowerBound( newVar );
        const double ub = engineQuery.getUpperBound( newVar );

        // If interval entirely >= 0 => Active
        if ( !FloatUtils::lt( lb, 0.0 ) )
            notifyLowerBoundUpdate(newVar, -INFINITY, lb);

        // If interval entirely <= 0 => Inactive
        else if ( !FloatUtils::gt( ub, 0.0 ) )
            notifyUpperBoundUpdate(newVar, +INFINITY, ub);
    }
}

void IncrementalConflictAnalyser::setNewEpsilon( double epsilon )
{
    _currentEpsilon = epsilon;
}

void IncrementalConflictAnalyser::addConflict(
    const std::vector<unsigned> &vars,
    const std::vector<bool> &isActiveList )
{
    ASSERT( _currentEpsilon >= 0.0 );
    ASSERT( vars.size() == isActiveList.size() );

    // Minimality pruning (optional; keep if it works)
    IncrementalConflictAnalyser::Bitmask subMask = _buildConflictSubBitmask( vars, isActiveList );
    if ( _isNonMinimalConflict( subMask ) )
        return;

    auto &bucket = _conflictsByEpsilon[_currentEpsilon];
    bucket.emplace_back( vars, isActiveList );
    // Encode immediately into current SAT solver (so future calls in this epsilon see it)
    _encodeConflictClause( bucket.back() );

    IncrementalConflictAnalyser::Bitmask fullMask = _buildConflictBitmask( vars, isActiveList );
    _minimalConflictBitmasks.push_back( fullMask );
}


void IncrementalConflictAnalyser::notifyNeuronFixed( unsigned newVar, ReLUState state )
{
    ASSERT( _seenPhase );
    ASSERT( _preprocessor );

    // engine new -> old var id
    const unsigned oldVar = _preprocessor->getOldIndex( newVar );

    ReLURuntimeState incoming =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Active : ReLURuntimeState::Inactive;

    const ReLURuntimeState opposite =
        ( state == ReLUState::Active ) ? ReLURuntimeState::Inactive : ReLURuntimeState::Active;

    // If both phases were seen across time, treat as Zero
    auto it = _seenPhase->find( oldVar );
    if ( it != _seenPhase->end() && ( *it ).second == opposite )
        incoming = ReLURuntimeState::Zero;

    // If present, must be the opposite (otherwise inconsistent)
    ASSERT( it == _seenPhase->end() || ( *it ).second == opposite );

    _seenPhase->insert( oldVar, incoming );
}


void IncrementalConflictAnalyser::notifyLowerBoundUpdate( unsigned newVar,
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
        notifyNeuronFixed( newVar, ReLUState::Active );
}

void IncrementalConflictAnalyser::notifyUpperBoundUpdate( unsigned newVar,
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
        notifyNeuronFixed( newVar, ReLUState::Inactive );
}


bool IncrementalConflictAnalyser::getImpliedTighteningsFromSat( List<Tightening> &tightenings )
{
    if ( !_conflictsExistForCurrent )
        return true;

    ASSERT( _currentEpsilon >= 0.0 );
    ASSERT( _preprocessor );
    ASSERT( _seenPhase );
    
    // Add assumptions for all currently fixed ReLUs in our SAT mapping.
    // IMPORTANT: ICA stores phases keyed by OLD vars, and the SAT mapping keys are OLD vars.
    for ( const auto &entry : _reluIndexToSatVarMap )
    {   
        const unsigned oldVar = entry.first;
        const ReLURuntimeState rt = _getReluPhase( oldVar );

        if ( rt == ReLURuntimeState::Active )
            _cadical->assume( _phaseToLit( oldVar, ReLUState::Active ) );
        else if ( rt == ReLURuntimeState::Inactive )
            _cadical->assume( _phaseToLit( oldVar, ReLUState::Inactive ) );
        // Zero/Unstable/Unseen: no assumption
    }

    // Propagate under assumptions
    const int res = _cadical->propagate();

    // 20 means conflict under assumptions
    if ( res == 20 )
        return false;

    // Query entailed literals (implications) — same as DA
    std::vector<int> implicants;
    _cadical->get_entrailed_literals( implicants );

    for ( int lit : implicants )
    {
        unsigned oldVar = 0;
        ReLUState impliedPhase;

        const bool success = _litToPhase( lit, oldVar, impliedPhase );
        ASSERT( success );

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
            ASSERT( false );
        }

        // Emit tightening in ENGINE var space (NEW var id)
        _emitTighteningsForImpliedPhase( oldVar, impliedPhase, tightenings );

    }

    return true;
}


ReLURuntimeState IncrementalConflictAnalyser::_getReluPhase( unsigned oldVar ) const
{
    ASSERT( _seenPhase );

    auto it = _seenPhase->find( oldVar );
    if ( it == _seenPhase->end() )
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

    ASSERT( _context );
    ASSERT( _preprocessor );
    ASSERT( !_conflictsExistForCurrent );

    // Epsilon must be valid
    ASSERT( _currentEpsilon >= 0.0 );
    if (_bitmaskSize == 0)
        _bitmaskSize = 2 * numQueryVariables + 3;

    // Bitmask must be large enough to index all variables
    ASSERT( _bitmaskSize >= 2 * numQueryVariables + 3 );

    if ( _reuseAllConflicts )
    {
        ASSERT (_minimalConflictBitmasks.size() == 0);
        _initializeSatSolver();
        _importRelevantConflicts();
    }

}


void IncrementalConflictAnalyser::notifySolved()
{

    _currentEpsilon = -1.0;
    _context = nullptr;
    _preprocessor = nullptr;
    _seenPhase = nullptr;
    _conflictsExistForCurrent = false;

    if ( _reuseAllConflicts )
    {
        _minimalConflictBitmasks.clear();
    }
}

void IncrementalConflictAnalyser::_initializeSatSolver()
{
    _cadical = std::make_unique<CaDiCaL::Solver>();

    _satVarToReluIndexMap.clear();
    _reluIndexToSatVarMap.clear();
    // Reserve index 0 so SAT vars 1..N map naturally to _satVarToReluIndexMap[1..N]
    _satVarToReluIndexMap.append( (unsigned)-1 );
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
    const unsigned newSatVar = _satVarToReluIndexMap.size();
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


void IncrementalConflictAnalyser::_encodeConflictClause( const Conflict &conflict )
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
        const unsigned satVar = _reluIndexToSatVarForce( oldVar );

        ASSERT( satVar > 0 );

        // Block the conflicting literal
        // If conflict says "active", we add ¬x; if "inactive", add x
        const int lit = active ? -(int)satVar : (int)satVar;

        _cadical->add( lit );
    }

    // Terminate clause
    _cadical->add( 0 );
    _conflictsExistForCurrent = true;

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
    
    size_t total = 0;
    for ( const auto &kv : _conflictsByEpsilon )
        total += kv.second.size();

    printf( "[ICA][IV] _importRelevantConflicts: currentEpsilon=%.6f, totalConflicts=%zu\n",
            _currentEpsilon,
            total );

    unsigned imported = 0;

    for ( auto it = _conflictsByEpsilon.lower_bound( _currentEpsilon );
        it != _conflictsByEpsilon.end(); ++it )
    {
        for ( const Conflict &conflict : it->second )
        {
            _encodeConflictClause( conflict );
            ++imported;
        }
    }

    printf( "[ICA][IV] _importRelevantConflicts done: imported=%u\n", imported );
}

