#include "IncrementalConflictAnalyser.h"
#include "Query.h"
#include "FloatUtils.h"
#include "DependencyCalculator.h"
#include <cmath> 

#include <cassert>

IncrementalConflictAnalyser::IncrementalConflictAnalyser( bool reuseAllConflicts )
    : _context( nullptr )
    , _preprocessor( nullptr )
    , _currentEpsilon( -1.0 )
    , _reuseAllConflicts( reuseAllConflicts )
    , _cadical( nullptr )
    , _bitmaskConflictSize( 0 )
    , _bitmaskDependencyVSize( 0 )
    , _seenPhase( nullptr )
    , _engineQuery( nullptr )
    , _nlr( nullptr )
    , _boundManager( nullptr )
    , _statistics( nullptr )
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

bool IncrementalConflictAnalyser::addConflict(
    const std::vector<unsigned> &vars,
    const std::vector<bool> &isActiveList )
{
    ASSERT( _currentEpsilon >= 0.0 );
    ASSERT( vars.size() == isActiveList.size() );

    // Minimality pruning (optional; keep if it works)
    DependencyAnalyzer::Bitmask subMask = _buildConflictSubBitmask( vars, isActiveList );
    if ( _isNonMinimalConflict( subMask ) )
        return false;

    _conflicts.emplace_back( _currentEpsilon, vars, isActiveList );

    // Encode immediately into current SAT solver (so future calls in this epsilon see it)
    _encodeConflictClause( _conflicts.back() );

    DependencyAnalyzer::Bitmask fullMask = _buildConflictBitmask( vars, isActiveList );
    _minimalConflictBitmasks.push_back( fullMask );
    return true;
}

void IncrementalConflictAnalyser::addDependency( const std::vector<unsigned> &vars,
                                                 const std::vector<bool> &isActiveList )
{
    ASSERT( _currentEpsilon >= 0.0 );
    ASSERT( vars.size() == isActiveList.size() );
    ASSERT( vars.size() > 0 );
    ASSERT( _bitmaskDependencyVSize > 0 );

    // Canonical ordering
    for ( size_t i = 1; i < vars.size(); ++i )
        ASSERT( vars[i - 1] < vars[i] );

    // --- Early pruning: vars only (ignore polarity) ---
    DependencyAnalyzer::Bitmask subMask = _buildDependencyVarsSubBitmask( vars );
    if ( _isNonMinimalDependencyVars( subMask ) )
        ASSERT( false && "Not supposed to happen.");

    // --- Store + encode as a conflict (this calls _encodeConflictClause) ---
    bool res = addConflict( vars, isActiveList );
    ASSERT(res);

    // --- Record minimal dependency var-set (full mask) ---
    // DependencyAnalyzer::Bitmask fullMask = _buildDependencyVarsBitmask( vars );
    _recordMinimalDependencyVars( vars );
}

bool IncrementalConflictAnalyser::isNonMinimalDependencyVarsSubMask(
    const std::vector<unsigned> &vars ) const
{
    ASSERT( _bitmaskDependencyVSize > 0 );

    // Empty set cannot be non-minimal
    if ( vars.empty() )
        return false;

    // Build vars-only *sub* mask (skips unmapped SAT vars)
    DependencyAnalyzer::Bitmask subMask( _bitmaskDependencyVSize );

    for ( unsigned oldVar : vars )
    {
        const unsigned satVar = _reluIndexToSatVar( oldVar );
        if ( satVar == 0 )
            continue; // unmapped → ignored in sub-mask

        ASSERT( satVar < _bitmaskDependencyVSize );
        subMask.set( satVar );
    }

    // If nothing mapped yet, cannot prune
    if ( subMask.none() )
        return false;

    // Superset check against known minimal dependency var-sets
    for ( const auto &known : _minimalDependencyVarBitmasks )
    {
        // known ⊆ subMask  → current vars are non-minimal
        if ( ( known & subMask ) == known )
            return true;
    }

    return false;
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
    if ( _conflicts.size() == 0 )
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

void IncrementalConflictAnalyser::notifySolvingStarted(
    unsigned numQueryVariables,
    const Query *engineQuery,
    NLR::NetworkLevelReasoner *nlr,
    BoundManager *boundManager,
    Statistics *statistics )
{
    ASSERT( _context );
    ASSERT( _preprocessor );

    printf(
        "[ICA] notifySolvingStarted: numVars=%u, eps=%.6f, reuseAll=%u\n",
        numQueryVariables,
        _currentEpsilon,
        _reuseAllConflicts );

    // --- Set per-solve state (must be null before) ---
    setEngineQuery( engineQuery );
    setNetworkLevelReasoner( nlr );
    setBoundManager( boundManager );
    setStatistics( statistics ); 

    ASSERT( _engineQuery );
    ASSERT( _nlr );
    ASSERT( _boundManager );
    ASSERT( _statistics );

    printf(
        "[ICA]  engineQuery=%p, nlr=%p, boundManager=%p\n",
        (void *)_engineQuery,
        (void *)_nlr,
        (void *)_boundManager );

    // Epsilon must be valid
    ASSERT( _currentEpsilon >= 0.0 );

    // --- Bitmask initialization ---
    if ( _bitmaskConflictSize == 0 )
    {
        _bitmaskConflictSize = 2 * numQueryVariables + 3;
        printf(
            "[ICA]  init conflict bitmask size = %u\n",
            _bitmaskConflictSize );
    }

    if ( _bitmaskDependencyVSize == 0 )
    {
        _bitmaskDependencyVSize = numQueryVariables + 1;
        printf(
            "[ICA]  init dependency-var bitmask size = %u\n",
            _bitmaskDependencyVSize );
    }

    ASSERT( _bitmaskConflictSize >= 2 * numQueryVariables + 3 );
    ASSERT( _bitmaskDependencyVSize >= numQueryVariables + 1 );

    // --- NLR sanity ---
    const NLR::NetworkLevelReasoner *fromQuery =
        _engineQuery->getNetworkLevelReasoner();
    ASSERT( fromQuery );
    ASSERT( _nlr );

    printf(
        "[ICA]  NLR check: fromQuery=%p, fromEngine=%p\n",
        (void *)fromQuery,
        (void *)_nlr );

    ASSERT( fromQuery == _nlr ); // expected to hold for now

    // --- Conflict reuse path ---
    if ( _reuseAllConflicts )
    {
        ASSERT( _minimalConflictBitmasks.empty() );
        printf( "[ICA]  initializing SAT solver + importing conflicts\n" );
        _initializeSatSolver();
        _importRelevantConflicts();
    }

    // --- Dependency discovery ---
    printf( "[ICA]  starting dependency calculation\n" );
    calculateDependencies();
    printf( "[ICA]  dependency calculation finished\n" );
    ASSERT( _statistics );

    // These are the dedicated fields you added in DependencyCalculator::Stats via Step 2,
    // but they need to be available here. Best: store them in ICA when calculateDependencies runs.
    // Assume ICA has members like: _ws1_unstable, _ws1_depsFound, _ws1_seconds, etc.

    _statistics->setUnsignedAttribute( Statistics::UNSTABLE_NEURONS_WS1, _ws1_unstable );
    _statistics->setUnsignedAttribute( Statistics::FOUND_DEPS_WS1, _ws1_depsFound );
    _statistics->setUnsignedAttribute( Statistics::UNSTABLE_NEURONS_WS2, _ws2_unstable );
    _statistics->setUnsignedAttribute( Statistics::FOUND_DEPS_WS2, _ws2_depsFound );

    _statistics->setDoubleAttribute( Statistics::SECONDS_TO_FIND_WS1_DEPS, _ws1_seconds );
    _statistics->setDoubleAttribute( Statistics::SECONDS_TO_FIND_WS2_DEPS, _ws2_seconds );

}


void IncrementalConflictAnalyser::calculateDependencies()
{
    ASSERT( _context );
    ASSERT( _preprocessor );

    // These must have been set by Engine right before notifySolvingStarted()
    ASSERT( _engineQuery );
    ASSERT( _nlr );
    ASSERT( _boundManager );

    DependencyCalculator calc( *this,
                               _engineQuery,
                               _nlr,
                               _preprocessor,
                               _boundManager );
    
    printf( "[ICA]  running DependencyCalculator\n" );
    calc.run();
    printf( "[ICA]  DependencyCalculator finished\n" );

    const auto &s = calc.getStats();

    _ws1_unstable = s.ws1_unstable;
    _ws1_depsFound = s.ws1_depsFound;
    _ws1_seconds = s.ws1_seconds;

    _ws2_unstable = s.ws2_unstable;
    _ws2_depsFound = s.ws2_depsFound;
    _ws2_seconds = s.ws2_seconds;
}

void IncrementalConflictAnalyser::setEngineQuery( const Query *q )
{
    // Must only be set once per solve
    ASSERT( _engineQuery == nullptr );
    ASSERT( q );

    _engineQuery = q;
}

void IncrementalConflictAnalyser::setNetworkLevelReasoner(
    NLR::NetworkLevelReasoner *nlr )
{
    // Must only be set once per solve
    ASSERT( _nlr == nullptr );
    ASSERT( nlr );

    _nlr = nlr;
}

void IncrementalConflictAnalyser::setBoundManager( BoundManager *bm )
{
    // Must only be set once per solve
    ASSERT( _boundManager == nullptr );
    ASSERT( bm );

    _boundManager = bm;
}

void IncrementalConflictAnalyser::setStatistics( Statistics *statistics )
{
    // Must only be set once per solve
    ASSERT( _statistics == nullptr );
    ASSERT( statistics );

    _statistics = statistics;
}


void IncrementalConflictAnalyser::notifySolved()
{
    _currentEpsilon = -1.0;
    _context = nullptr;
    _preprocessor = nullptr;
    _seenPhase = nullptr;

    _engineQuery = nullptr;
    _nlr = nullptr;
    _boundManager = nullptr;

    if ( _reuseAllConflicts )
    {
        _minimalConflictBitmasks.clear();
        _minimalDependencyVarBitmasks.clear();
    }
    
    _statistics = nullptr;
    _ws1_unstable = _ws1_depsFound = 0;
    _ws1_seconds = 0.0;
    _ws2_unstable = _ws2_depsFound = 0;
    _ws2_seconds = 0.0;

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
    const DependencyAnalyzer::Bitmask &mask ) const
{
    for ( const auto &known : _minimalConflictBitmasks )
    {
        if ( ( known & mask ) == known )
            return true;
    }
    return false;
}


DependencyAnalyzer::Bitmask
IncrementalConflictAnalyser::_buildConflictBitmask( const std::vector<unsigned> &vars,
                                                   const std::vector<bool> &isActive ) const
{
    ASSERT( vars.size() == isActive.size() );
    DependencyAnalyzer::Bitmask mask( _bitmaskConflictSize );

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

DependencyAnalyzer::Bitmask
IncrementalConflictAnalyser::_buildConflictSubBitmask( const std::vector<unsigned> &vars,
                                                      const std::vector<bool> &isActive ) const
{
    ASSERT( vars.size() == isActive.size() );
    DependencyAnalyzer::Bitmask mask( _bitmaskConflictSize );

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

// Used by DependencyCalculator BEFORE doing expensive analysis.
// Prunes supersets of already-known minimal dependency var-sets.
//
// Semantics (vars-only):
//   Let mask represent a candidate dependency variable set (SAT-var indexed).
//   If we already recorded some known minimal var-set K such that K ⊆ mask,
//   then this candidate is a superset and we should skip exploring it.
bool IncrementalConflictAnalyser::_isNonMinimalDependencyVars(
    const DependencyAnalyzer::Bitmask &mask ) const
{
    for ( const auto &known : _minimalDependencyVarBitmasks )
    {
        // If known ⊆ mask, then (known & mask) == known
        if ( ( known & mask ) == known )
            return true;
    }
    return false;
}

DependencyAnalyzer::Bitmask
IncrementalConflictAnalyser::_buildDependencyVarsBitmask(
    const std::vector<unsigned> &vars ) const
{
    ASSERT( _bitmaskDependencyVSize > 0 );

    DependencyAnalyzer::Bitmask mask( _bitmaskDependencyVSize );

    for ( unsigned oldVar : vars )
    {
        const unsigned satVar = _reluIndexToSatVar( oldVar );
        ASSERT( satVar != 0 ); // full mask requires full SAT mapping

        ASSERT( satVar < mask.size() );
        mask.set( satVar );
    }

    return mask;
}


DependencyAnalyzer::Bitmask
IncrementalConflictAnalyser::_buildDependencyVarsSubBitmask(
    const std::vector<unsigned> &vars ) const
{
    ASSERT( _bitmaskDependencyVSize > 0 );

    DependencyAnalyzer::Bitmask mask( _bitmaskDependencyVSize );

    for ( unsigned oldVar : vars )
    {
        const unsigned satVar = _reluIndexToSatVar( oldVar );

        // Sub-mask: skip vars that are not yet mapped to SAT
        if ( satVar == 0 )
            continue;

        ASSERT( satVar < mask.size() );
        mask.set( satVar );
    }

    return mask;
}

void IncrementalConflictAnalyser::_recordMinimalDependencyVars(
    const std::vector<unsigned> &vars )
{
    ASSERT( _bitmaskDependencyVSize > 0 );
    ASSERT( vars.size() > 0 );

    // Build full dependency-var bitmask (vars only, no polarity)
    DependencyAnalyzer::Bitmask mask( _bitmaskDependencyVSize );

    for ( unsigned oldVar : vars )
    {
        // This assumes we call _reluIndexToSatVarForce(oldVar) before recording.
        const unsigned satVar = _reluIndexToSatVar( oldVar );
        ASSERT( satVar != 0 ); // must be mapped before recording

        ASSERT( satVar < mask.size() );
        mask.set( satVar );
    }

    _minimalDependencyVarBitmasks.push_back( std::move( mask ) );
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
    // printf( "[ICA][dep-min] oldVar=%u -> satVar=%u\n",
    //         relu, newSatVar );
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
    printf( "[ICA][IV] _importRelevantConflicts: currentEpsilon=%.6f, totalConflicts=%zu\n",
            _currentEpsilon,
            _conflicts.size() );

    unsigned imported = 0;

    for ( const Conflict &conflict : _conflicts )
    {
        const double eps = conflict.getEpsilon();

        if ( eps >= _currentEpsilon )
        {
            _encodeConflictClause( conflict );
            _encodeMinimalBitmasks( conflict );
            ++imported;
        }
        else
        {
            // printf( "[ICA][IV]   skipping conflict (eps=%.6f < current)\n", eps );
        }
    }

    printf( "[ICA][IV] _importRelevantConflicts done: imported=%u\n", imported );
}

void IncrementalConflictAnalyser::_encodeMinimalBitmasks( const Conflict &conflict )
{
    ASSERT( _bitmaskConflictSize > 0 );
    ASSERT( _bitmaskDependencyVSize > 0 );

    const auto &vars     = conflict.getVars();
    const auto &isActive = conflict.getIsActive();

    ASSERT( vars.size() == isActive.size() );

    if ( vars.empty() )
        return;

    // --------------------------------------------------
    // 1) Conflict minimal bitmask (polarity-aware)
    // --------------------------------------------------
    {
        // Sub-mask first (safe even if something odd happens)
        DependencyAnalyzer::Bitmask subMask =
            _buildConflictSubBitmask( vars, isActive );

        // ASSERT ( !_isNonMinimalConflict( subMask ) );
        if ( !_isNonMinimalConflict( subMask ) )
        {
            DependencyAnalyzer::Bitmask fullMask =
                _buildConflictBitmask( vars, isActive );

            _minimalConflictBitmasks.push_back( fullMask );
        }
    }

    // --------------------------------------------------
    // 2) Dependency-vars minimal bitmask (vars-only)
    // --------------------------------------------------
    {
        // Build vars-only *sub* mask for pruning
        DependencyAnalyzer::Bitmask subVarMask( _bitmaskDependencyVSize );

        for ( unsigned oldVar : vars )
        {
            const unsigned satVar = _reluIndexToSatVar( oldVar );
            ASSERT( satVar != 0 ); // must exist after _encodeConflictClause
            ASSERT( satVar < _bitmaskDependencyVSize );
            subVarMask.set( satVar );
        }

        // If not subsumed by an existing minimal var-set, record it
        // ASSERT( !_isNonMinimalDependencyVars( subVarMask ) );
        if ( !subVarMask.none() && !_isNonMinimalDependencyVars( subVarMask ) )
        {
            // IMPORTANT: record by vars, not by mask
            _recordMinimalDependencyVars( vars );
        }
    }
}
